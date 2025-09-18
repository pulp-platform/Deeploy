# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
import math
from typing import List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AnydimAsyncDmaTransferAdapter, AsyncDma, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme, stridesFromShape


class DoubleBufferingTilingCodeGeneration(TilingCodeGeneration):

    _moveTileInCheckOpenStatement = NodeTemplate("""
    // DOUBLE BUFFERING CHECK TILE LOAD
    if ((${tileIdxVar}) < ${numTiles}[*${tileIdxPtr}+1]) {
    """)

    _moveTileInCheckCloseStatement = NodeTemplate("""
    }
    """)

    # LMACAN: The brackets around ${tileIdxVar} are important to ensure correct order
    #         of the modulo operation. Breaking case without the brackets is when we
    #         put "TILING_I + 1" for tileIdxVar.
    _chooseBufferTemplate = NodeTemplate("""
    switch((${tileIdxVar}) % 2) {
        case 0: ${reference} = (${type})${buffer_0}; break;
        case 1: ${reference} = (${type})${buffer_1}; break;
    }
    """)

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 2)

    def _generateBufferChoice(self, reference: VariableBuffer, buffers: List[_ReferenceBuffer],
                              tileIdxVar: str) -> CodeSnippet:
        assert len(buffers) == 2, f"Only double buffering supported. Received {len(buffers)} buffers."
        operatorRepresentation = {
            "tileIdxVar": tileIdxVar,
            "reference": reference.name,
            "type": reference._type.typeName,
            "buffer_0": buffers[0].name,
            "buffer_1": buffers[1].name,
        }
        template = self._chooseBufferTemplate
        return CodeSnippet(template, operatorRepresentation)

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        setupStatements: List[CodeSnippet] = []
        teardownStatements: List[CodeSnippet] = []

        openLoopStatements: List[CodeSnippet] = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        ingressDmaTransferCalls: List[CodeSnippet] = [
            CodeSnippet(self._moveTileInCheckOpenStatement, {
                **operatorRepresentation, "tileIdxVar": "TILING_I+1"
            })
        ]

        ingressFutures: Set[Future] = set()
        initialFutures: Set[Future] = set()

        for tensorName, rectangles in dictOfArrays(tilingSchedule.inputLoadSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[externalBuffer.name]
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

            tensorMemoryConstraint = nodeMemoryConstraint.inputTensorMemoryConstraints[externalBuffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)

            nextLocalBufferReference = self._hoistReference(ctxt, f"{tensorName}_next", l1BuffersReferences[1])

            openLoopStatements.append(self._generateBufferChoice(localBuffer, l1BuffersReferences, "TILING_I"))

            future = self.dma.getFuture(tensorName)
            ingressFutures.add(future)

            ingressDmaTransferCalls.append(
                self._generateBufferChoice(nextLocalBufferReference, l1BuffersReferences, "TILING_I+1"))
            ingressDmaTransferCalls.extend(
                self._generateDmaTransferCalls(ctxt, tensorName, rectangles, "TILING_I+1", nextLocalBufferReference,
                                               externalBufferRef, "ExternalToLocal", future))

            anydimAdapter = AnydimAsyncDmaTransferAdapter(self.dma)

            initialFuture = self.dma.getFuture(tensorName + "_init")
            initialFutures.add(initialFuture)
            initialDmaTransferCalls = anydimAdapter.transfer(ctxt, externalBufferRef, localBuffer, rectangles[0].dims,
                                                             stridesFromShape(externalBufferShape),
                                                             stridesFromShape(rectangles[0].dims), "ExternalToLocal",
                                                             initialFuture, math.prod(externalBufferShape))
            setupStatements.extend(initialDmaTransferCalls)
            setupStatements.append(initialFuture.wait())

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I+1",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                ingressDmaTransferCalls.append(referenceUpdate)
                initialReferenceUpdate = CodeSnippet(referenceUpdate.template,
                                                     operatorRepresentation = {
                                                         **referenceUpdate.operatorRepresentation,
                                                         "tileIdxVar": 0,
                                                     })
                setupStatements.append(initialReferenceUpdate)

        ingressDmaTransferCalls.append(CodeSnippet(self._moveTileInCheckCloseStatement, {}))
        ingressDmaWaitStatements = [f.wait() for f in ingressFutures]

        egressDmaTransferCalls: List[CodeSnippet] = []
        egressFutures: Set[Future] = set()

        for tensorName, rectangles in dictOfArrays(tilingSchedule.outputLoadSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[externalBuffer.name]
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

            tensorMemoryConstraint = nodeMemoryConstraint.outputTensorMemoryConstraints[externalBuffer.name]
            l1BuffersReferences = self._hoistMultibufferReferences(ctxt, localBuffer, tensorMemoryConstraint)

            openLoopStatements.append(self._generateBufferChoice(localBuffer, l1BuffersReferences, "TILING_I"))

            future = self.dma.getFuture(tensorName)
            egressFutures.add(future)

            dmaTransferCalls = self._generateDmaTransferCalls(ctxt, tensorName, rectangles, "TILING_I", localBuffer,
                                                              externalBufferRef, "LocalToExternal", future)
            egressDmaTransferCalls.extend(dmaTransferCalls)

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, "TILING_I",
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                egressDmaTransferCalls.append(referenceUpdate)

        egressDmaWaitStatements = [f.wait() for f in egressFutures]

        teardownStatements.extend([f.wait() for f in egressFutures])

        setupStatements = [f.init() for f in ingressFutures | initialFutures | egressFutures] + setupStatements
        teardownStatements.extend(f.deinit() for f in ingressFutures | initialFutures | egressFutures)

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
