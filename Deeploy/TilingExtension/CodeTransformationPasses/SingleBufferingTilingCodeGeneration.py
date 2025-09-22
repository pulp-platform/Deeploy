# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Set, Tuple

from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation, \
    VariableBuffer, _ReferenceBuffer
from Deeploy.TilingExtension.AsyncDma import AsyncDma, DmaDirection, Future
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingHoistingMixIn import dictOfArrays
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import ProfilingPrototypeMixIn, \
    PrototypeTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme

_measureConditionSetup = NodeTemplate("""
if(${cond}){
""")

_measureConditionEnd = NodeTemplate("""
}
""")

_measureCycles = NodeTemplate("""
${measurements}[${tileIdxVar}] = getCycles();
""")


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

        # Single Buffering Tiling Loop Strategy
        # ===================================
        # - 1) Initialize all futures
        # - 2) for TILING_I in numTiles:
        #   - 2.1) Input data transfer for current tile (see "4.2) Input Data Transfers")
        #   - 2.2) Process current tile
        #   - 2.3) Output data transfer for current tile (see "4.4) Output Data Transfers")
        # - 3) Deinitialize all futures

        # 2) for TILING_I in numTiles:
        openLoopStatements = [CodeSnippet(self._openTileLoopTemplate, {**operatorRepresentation})]

        # 2.2) Input data transfer for current tile
        ctxt, ingressDMAStatements, ingressFutures = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
            nodeMemoryConstraint.inputTensorMemoryConstraints, "TILING_I", "ExternalToLocal", "Transfer input tile")

        ingressDMAStatements += [future.wait("Wait for input tile") for future in ingressFutures]

        # 2.4) Output data transfer for current tile
        ctxt, egressDMAStatements, egressFutures = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
            nodeMemoryConstraint.outputTensorMemoryConstraints, "TILING_I", "LocalToExternal", "Transfer output tile")
        egressDMAStatements += [future.wait("Wait for output tile") for future in egressFutures]

        # 1) Initialize all futures
        setupStatements = [f.init("Initialize DMA future") for f in ingressFutures | egressFutures]

        # 3) Deinitialize all futures
        teardownStatements = [f.deinit("Deinitialize DMA future") for f in ingressFutures | egressFutures]

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

        executionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDMAStatements, egressDMAStatements,
                                                    openLoopStatements, closeLoopStatements, setupStatements,
                                                    teardownStatements)

        return ctxt, executionBlock, True


class ProfilingSingleBufferingTilingMixIn(PrototypeTilingMixIn, ProfilingPrototypeMixIn):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: List[CodeSnippet],
                                     teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)

        executionBlock = cls.measurementArrayDeclaration(executionBlock, metaInfo, bufferingStr = "SB")

        executionBlock = cls.injectPrintCycleDiff(executionBlock, metaInfo)

        return executionBlock

    @classmethod
    def generateLoopCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                         openLoopStatements: List[CodeSnippet], ingressDMAStatements: List[CodeSnippet],
                         egressDMAStatements: List[CodeSnippet],
                         closeLoopStatements: List[CodeSnippet]) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        _openLoopStatements = [openLoopStatements[0]]
        _openLoopStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _openLoopStatements += openLoopStatements[1:]

        _ingressDMAStatements = []
        _ingressDMAStatements += ingressDMAStatements
        _ingressDMAStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = cls.kernelProfilingWrap(executionBlock, metaInfo)

        _egressDMAStatements = []
        _egressDMAStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _egressDMAStatements += egressDMAStatements
        _egressDMAStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = super().generateLoopCode(executionBlock, metaInfo, _openLoopStatements, _ingressDMAStatements,
                                                  _egressDMAStatements, closeLoopStatements)
        return executionBlock
