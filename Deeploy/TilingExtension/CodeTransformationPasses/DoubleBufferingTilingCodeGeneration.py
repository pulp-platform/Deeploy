# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
import itertools
from typing import Dict, List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme


class DoubleBufferingTilingCodeGeneration(TilingCodeGeneration):

    _moveTileInCheckOpenStatement = NodeTemplate("""
    // DOUBLE BUFFERING CHECK TILE LOAD
    if ((${tileIdxVar}) < ${numTiles}[*${tileIdxPtr}+1]) {
    """)

    # LMACAN: The brackets around ${tileIdxVar} are important to ensure correct order
    #         of the modulo operation. Breaking case without the brackets is when we
    #         put "TILING_I + 1" for tileIdxVar.
    _switchOpen = NodeTemplate("switch((${tileIdxVar}) % ${bufferCount}) {")
    _caseOpen = NodeTemplate("case ${case}:")
    _caseClose = NodeTemplate("break;")

    _blockClose = NodeTemplate("""
    }
    """)

    _referenceUpdate = NodeTemplate("${reference} = (${type})${update};")

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 2)

    def _switch(self, caseBlocks: List[List[CodeSnippet]], tileIdxVar: str) -> List[CodeSnippet]:
        assert len(caseBlocks) == self.bufferCount
        callStack = [CodeSnippet(self._switchOpen, {"tileIdxVar": tileIdxVar, "bufferCount": self.bufferCount})]
        for i, block in enumerate(caseBlocks):
            callStack.append(CodeSnippet(self._caseOpen, {"case": i}))
            callStack.extend(block)
            callStack.append(CodeSnippet(self._caseClose, {}))
        callStack.append(CodeSnippet(self._blockClose, {}))
        return callStack

    def _generateBufferChoice(self, reference: VariableBuffer, buffers: List[_ReferenceBuffer],
                              tileIdxVar: str) -> List[CodeSnippet]:
        return self._switch([[
            CodeSnippet(self._referenceUpdate, {
                "reference": reference.name,
                "type": reference._type.typeName,
                "update": buff.name
            })
        ] for buff in buffers], tileIdxVar)

    def _ioCalls(
        self,
        ctxt: NetworkContext,
        operatorRepresentation: OperatorRepresentation,
        schedule: List[Dict[str, HyperRectangle]],
        memoryConstraints: Dict[str, TensorMemoryConstraint],
        direction: DmaDirection,
        tileIdxVar: str,
        multibufferMap: Dict[str, List[_ReferenceBuffer]],
    ) -> Tuple[List[CodeSnippet], Set[Future]]:
        calls, futures = [], set()
        for tensorName, rectangles in dictOfArrays(schedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = memoryConstraints[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            rectangles, externalBufferShape = self._legalizeTransfers(rectangles, tuple(externalBufferShape),
                                                                      localBuffer._type.referencedType.typeWidth,
                                                                      self.isFinalMemoryLevel(tensorMemoryConstraint))

            externalBufferRef = self._hoistReference(ctxt,
                                                     externalBuffer.name + "_ref",
                                                     externalBuffer,
                                                     externalBufferShape,
                                                     override_type = VoidType)

            caseBlocks = []
            for i, buff in enumerate(multibufferMap[tensorName]):
                future = self.dma.getFuture(tensorName, copyIdx = i)
                futures.add(future)

                block = [future.alloc()]
                block.extend(
                    self._generateDmaTransferCalls(ctxt, tensorName, rectangles, tileIdxVar, buff, externalBufferRef,
                                                   direction, future))
                caseBlocks.append(block)

            calls.extend(self._switch(caseBlocks, tileIdxVar))

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, tileIdxVar,
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                calls.append(referenceUpdate)

        return calls, futures

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        multibufferMap: Dict[str, List[_ReferenceBuffer]] = {}

        for name in tilingSchedule.inputLoadSchedule[0].keys():
            localBuffer = ctxt.lookup(operatorRepresentation[name])
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[externalBuffer.name]
            buffers = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)
            multibufferMap[name] = buffers

        for name in tilingSchedule.outputLoadSchedule[0].keys():
            localBuffer = ctxt.lookup(operatorRepresentation[name])
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[externalBuffer.name]
            buffers = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)
            multibufferMap[name] = buffers

        openLoopStatements: List[CodeSnippet] = [CodeSnippet(self._openTileLoopTemplate, operatorRepresentation)]

        bufferGroupChoices = [[] for _ in range(self.bufferCount)]
        for name in itertools.chain(tilingSchedule.inputLoadSchedule[0].keys(),
                                    tilingSchedule.outputLoadSchedule[0].keys()):
            localBuffer = ctxt.lookup(operatorRepresentation[name])
            assert isinstance(localBuffer, _ReferenceBuffer)
            for i, buffer in enumerate(multibufferMap[name]):
                snippet = CodeSnippet(self._referenceUpdate, {
                    "reference": localBuffer.name,
                    "type": localBuffer._type.typeName,
                    "update": buffer.name
                })
                bufferGroupChoices[i].append(snippet)

        openLoopStatements += self._switch(bufferGroupChoices, "TILING_I")

        ingressCalls, ingressFutures = self._ioCalls(ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
                                                     nodeMemoryConstraint.inputTensorMemoryConstraints,
                                                     "ExternalToLocal", "TILING_I+1", multibufferMap)

        ingressDmaTransferCalls: List[CodeSnippet] = [
            CodeSnippet(self._moveTileInCheckOpenStatement, {
                **operatorRepresentation, "tileIdxVar": "TILING_I+1"
            })
        ] + ingressCalls + [CodeSnippet(self._blockClose, {})]
        ingressDmaWaitStatements = [f.wait() for f in ingressFutures]

        firstIngressCalls = []
        for snippet in ingressCalls:
            tmpl, opRepr = snippet.template, snippet.operatorRepresentation
            firstIngressCalls.append(CodeSnippet(tmpl, {**opRepr, "tileIdxVar": 0}))

        egressCalls, egressFutures = self._ioCalls(ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
                                                   nodeMemoryConstraint.outputTensorMemoryConstraints,
                                                   "LocalToExternal", "TILING_I", multibufferMap)

        egressDmaTransferCalls: List[CodeSnippet] = egressCalls
        egressDmaWaitStatements = [f.wait() for f in egressFutures]

        setupStatements: List[CodeSnippet] = []
        setupStatements += [f.init() for f in ingressFutures | egressFutures] + setupStatements
        setupStatements.extend(firstIngressCalls)

        teardownStatements: List[CodeSnippet] = []
        teardownStatements.extend([f.wait() for f in egressFutures])
        teardownStatements.extend(f.deinit() for f in ingressFutures | egressFutures)

        closeLoopStatements = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]

        metaInfo = TilingMetaInfo(
            nodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}",
            nodeOps = operatorRepresentation['nodeOps'],
            numTiles = operatorRepresentation['numTiles'],
            totalNumTiles = len(tilingSchedule.outputLoadSchedule),
            tileIdxPtr = operatorRepresentation['tileIdxPtr'],
            tileIdxVar = "TILING_I",
            # TODO: The kernelLevelTiling field is used in profiling to know we are generating code around the kernel.
            #       The current implementation does this by checking whether we are at the lowest memory level,
            #       which is hardcoded by the value "L1". Change this to be memory level agnostic.
            kernelLevelTiling = self.localMemory == "L1")

        executionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDmaTransferCalls,
                                                    ingressDmaWaitStatements, [], egressDmaTransferCalls,
                                                    egressDmaWaitStatements, [], [], openLoopStatements,
                                                    closeLoopStatements, setupStatements, teardownStatements)

        return ctxt, executionBlock, True

    def generateTilingLoop(
            self, ctxt: NetworkContext, executionBlock: ExecutionBlock, nodeMemoryConstraint: NodeMemoryConstraint,
            tilingSchedules: List[TilingSchedule], variableReplacement: VariableReplacementScheme,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        flatTilingSchedule = copy.copy(tilingSchedules[0])
        for tilingSchedule in tilingSchedules[1:]:
            flatTilingSchedule += tilingSchedule

        offsetLists = list({**flatTilingSchedule.inputBaseOffsets, **flatTilingSchedule.outputBaseOffsets}.values())

        if len(offsetLists) == 0:
            return ctxt, executionBlock, False

        for offsetList in offsetLists:
            if not len(offsetList) == self.bufferCount:
                return ctxt, executionBlock, False

        numTiles, tileIdxPtr = self._hoistTileNumAndIdxPtr(ctxt, tilingSchedules)
        operatorRepresentation["numTiles"] = numTiles.name
        operatorRepresentation["tileIdxPtr"] = tileIdxPtr.name

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)
