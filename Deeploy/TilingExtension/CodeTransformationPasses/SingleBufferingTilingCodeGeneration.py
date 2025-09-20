# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Dict, List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, OperatorRepresentation, VariableBuffer, \
    _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme


class SingleBufferingTilingCodeGeneration(TilingCodeGeneration):

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 1)

    def _generateTransferScheduleCalls(self,
                                       ctxt: NetworkContext,
                                       operatorRepresentation: OperatorRepresentation,
                                       transferSchedule: List[Dict[str, HyperRectangle]],
                                       tensorMemoryConstraintDict: Dict[str, TensorMemoryConstraint],
                                       tileIdxVar: str,
                                       direction: DmaDirection,
                                       comment: str = "") -> Tuple[NetworkContext, List[CodeSnippet], Set[Future]]:
        callStack: List[CodeSnippet] = []
        referenceUpdates: List[CodeSnippet] = []
        futures: Set[Future] = set()

        for tensorName, rectangles in dictOfArrays(transferSchedule).items():
            localBuffer = ctxt.lookup(operatorRepresentation[tensorName])
            assert localBuffer._memoryLevel == self.localMemory
            assert isinstance(localBuffer, _ReferenceBuffer)
            externalBuffer = ctxt.lookup(localBuffer._referenceName)
            assert isinstance(externalBuffer, VariableBuffer)
            tensorMemoryConstraint = tensorMemoryConstraintDict[externalBuffer.name]
            externalBufferShape = tensorMemoryConstraint.memoryConstraints[self.externalMemory].shape
            assert externalBufferShape is not None

            rectangles, externalBufferShape = self._legalizeTransfers(rectangles, tuple(externalBufferShape),
                                                                      localBuffer._type.referencedType.typeWidth,
                                                                      self.isFinalMemoryLevel(tensorMemoryConstraint))

            externalBufferRef = self._hoistReference(ctxt,
                                                     externalBuffer.name + "_ref",
                                                     externalBuffer,
                                                     shape = externalBufferShape,
                                                     override_type = VoidType)

            future = self.dma.getFuture(tensorName, direction)
            futures.add(future)

            callStack.extend(
                self._generateDmaTransferCalls(ctxt,
                                               tensorName,
                                               rectangles,
                                               tileIdxVar,
                                               localBuffer,
                                               externalBufferRef,
                                               direction,
                                               future,
                                               comment = comment))

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, tileIdxVar,
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                callStack.append(referenceUpdate)

        return ctxt, callStack, futures

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:
        ctxt, ingressDmaTransferCalls, ingressFutures = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
            nodeMemoryConstraint.inputTensorMemoryConstraints, "TILING_I", "ExternalToLocal", "Transfer input tile")
        ctxt, egressDmaTransferCalls, egressFutures = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
            nodeMemoryConstraint.outputTensorMemoryConstraints, "TILING_I", "LocalToExternal", "Transfer output tile")

        ingressDmaWaitStatements = [future.wait("Wait for input tile") for future in ingressFutures]
        egressDmaWaitStatements = [future.wait("Wait for output tile") for future in egressFutures]

        setupStatements = [f.init("Initialize DMA future") for f in ingressFutures | egressFutures]
        teardownStatements = [f.deinit("Deinitialzie DMA future") for f in ingressFutures | egressFutures]

        openLoopStatements = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]
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
