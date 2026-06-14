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
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import PerfCounterProfilingMixIn, \
    ProfilingPrototypeMixIn, PrototypeTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint, TensorMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme


class SingleBufferingTilingCodeGeneration(TilingCodeGeneration):

    # DEBUG: live progress beacons, emitted UNCONDITIONALLY (independent of
    # --profileTiling), so the normal build prints where it is. The last beacon
    # on the UART before a hang pinpoints the node + sub-phase that deadlocked.
    # NOTE: intentionally UNGUARDED (no `if (pi_core_id() == 0)`). The closure
    # setup/ingress code runs on a single core whose id is NOT 0, so a `==0`
    # guard suppresses every beacon (which is why the profiling beacons never
    # printed). The unguarded profiling SUMMARY prints DO appear, so a bare
    # printf here matches that and fires exactly once per phase.
    _traceNodeBeacon = NodeTemplate("""
    printf("[TRACE-NODE] ${nodeName}: ${phase}\\r\\n");
    """)
    _traceBeacon = NodeTemplate("""
    printf("[TRACE] ${nodeName} tile %u: ${phase}\\r\\n", ${tileIdxVar});
    """)

    def __init__(self, externalMemory: str, localMemory: str, dma: AsyncDma):
        super().__init__(externalMemory, localMemory, dma, 1)

    def _generateTransferScheduleCalls(
            self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
            transferSchedule: List[Dict[str, HyperRectangle]], tensorMemoryConstraintDict: Dict[str,
                                                                                                TensorMemoryConstraint],
            tileIdxVar: str, direction: DmaDirection) -> Tuple[NetworkContext, List[CodeSnippet], Set[Future]]:
        callStack: List[CodeSnippet] = []
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

            # Allocate a future for this transfer
            if future not in futures:
                callStack.append(future.alloc())

            try:
                callStack.extend(
                    self._generateDmaTransferCalls(ctxt, tensorName, rectangles, tileIdxVar, localBuffer,
                                                   externalBufferRef, direction, future))
            except AssertionError as e:
                raise AssertionError(f"{e} while generating DMA transfer for tensor '{tensorName}'") from e

            referenceUpdate = self._generateExternalReferenceUpdate(ctxt, tensorName, rectangles, tileIdxVar,
                                                                    externalBufferRef)
            if referenceUpdate is not None:
                callStack.append(referenceUpdate)

            futures.add(future)

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
            nodeMemoryConstraint.inputTensorMemoryConstraints, "TILING_I", "ExternalToLocal")

        ingressDMAStatements = [CodeSnippet(self._lineComment, {"comment": "Transfer input tiles"})
                               ] + ingressDMAStatements
        ingressDMAStatements += [CodeSnippet(self._lineComment, {"comment": "Wait for input tiles"})]
        ingressDMAStatements += [future.wait() for future in ingressFutures]

        # 2.4) Output data transfer for current tile
        ctxt, egressDMAStatements, egressFutures = self._generateTransferScheduleCalls(
            ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
            nodeMemoryConstraint.outputTensorMemoryConstraints, "TILING_I", "LocalToExternal")
        egressDMAStatements = [CodeSnippet(self._lineComment, {"comment": "Transfer output tiles"})
                              ] + egressDMAStatements
        egressDMAStatements += [CodeSnippet(self._lineComment, {"comment": "Wait for output tiles"})]
        egressDMAStatements += [future.wait() for future in egressFutures]

        # 1) Initialize all futures
        setupStatements = [CodeSnippet(self._lineComment, {"comment": "Initialize DMA futures"})]
        setupStatements.extend([f.init() for f in ingressFutures | egressFutures])

        # 3) Deinitialize all futures
        teardownStatements = [CodeSnippet(self._lineComment, {"comment": "Deinitialize DMA futures"})]
        teardownStatements.extend([f.deinit() for f in ingressFutures | egressFutures])

        closeLoopStatements = [CodeSnippet(self._closeTileLoopTemplate, {**operatorRepresentation})]

        # DEBUG: unconditional live beacons bracketing each phase. "ingress done"
        # is appended AFTER the future.wait()s, so if it never prints the hang is
        # in the L2<-external DMA wait of that tile.
        _beaconNodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}"
        setupStatements.insert(
            0, CodeSnippet(self._traceNodeBeacon, {
                "nodeName": _beaconNodeName,
                "phase": "ENTER (setup/alloc)"
            }))
        ingressDMAStatements.insert(
            0, CodeSnippet(self._traceBeacon, {
                "nodeName": _beaconNodeName,
                "phase": "ingress DMA start",
                "tileIdxVar": "TILING_I"
            }))
        ingressDMAStatements.append(
            CodeSnippet(self._traceBeacon, {
                "nodeName": _beaconNodeName,
                "phase": "ingress done -> kernel start",
                "tileIdxVar": "TILING_I"
            }))
        egressDMAStatements.insert(
            0, CodeSnippet(self._traceBeacon, {
                "nodeName": _beaconNodeName,
                "phase": "kernel done -> egress DMA start",
                "tileIdxVar": "TILING_I"
            }))
        egressDMAStatements.append(
            CodeSnippet(self._traceBeacon, {
                "nodeName": _beaconNodeName,
                "phase": "egress done (tile complete)",
                "tileIdxVar": "TILING_I"
            }))

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + f"_{self.externalMemory}",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = operatorRepresentation['numTiles'],
                                  totalNumTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxPtr = operatorRepresentation['tileIdxPtr'],
                                  tileIdxVar = "TILING_I",
                                  kernelLevelTiling = True)

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

        # addLeft last => frontmost statement of the node: fires before DMA-future
        # init / L3 alloc, so a setup-phase hang is still attributed to this node.
        executionBlock.addLeft(cls._liveNodeBeacon, {"nodeName": metaInfo.nodeName, "phase": "ENTER (setup/alloc)"})

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
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _openLoopStatements.append(
            CodeSnippet(cls._liveBeacon, {
                "nodeName": nodeName,
                "phase": "ingress DMA start",
                "tileIdxVar": tileIdxVar
            }))
        _openLoopStatements += openLoopStatements[1:]

        _ingressDMAStatements = []
        _ingressDMAStatements += ingressDMAStatements
        _ingressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _ingressDMAStatements.append(
            CodeSnippet(cls._liveBeacon, {
                "nodeName": nodeName,
                "phase": "ingress done -> kernel start",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = cls.kernelProfilingWrap(executionBlock, metaInfo)

        _egressDMAStatements = []
        _egressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _egressDMAStatements.append(
            CodeSnippet(cls._liveBeacon, {
                "nodeName": nodeName,
                "phase": "kernel done -> egress DMA start",
                "tileIdxVar": tileIdxVar
            }))
        _egressDMAStatements += egressDMAStatements
        _egressDMAStatements.append(
            CodeSnippet(cls._measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _egressDMAStatements.append(
            CodeSnippet(cls._liveBeacon, {
                "nodeName": nodeName,
                "phase": "egress done (tile complete)",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = super().generateLoopCode(executionBlock, metaInfo, _openLoopStatements, _ingressDMAStatements,
                                                  _egressDMAStatements, closeLoopStatements)
        return executionBlock


class PerfCounterSingleBufferingTilingMixIn(PrototypeTilingMixIn, PerfCounterProfilingMixIn):
    """
    Single buffering tiling with performance counter profiling.
    Provides detailed instruction-level statistics for each tile.
    """

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: List[CodeSnippet],
                                     teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)

        # Inject performance counter initialization in setup (only once, not per-tile)
        executionBlock = cls.injectPerfCounterInit(executionBlock, metaInfo)

        # Inject performance counter stop and print in teardown (only once, not per-tile)
        executionBlock = cls.injectPerfCounterStop(executionBlock, metaInfo)

        return executionBlock

    @classmethod
    def generateLoopCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                         openLoopStatements: List[CodeSnippet], ingressDMAStatements: List[CodeSnippet],
                         egressDMAStatements: List[CodeSnippet],
                         closeLoopStatements: List[CodeSnippet]) -> ExecutionBlock:

        # Don't wrap kernel - perf counters measure the whole tiling loop, not individual tiles
        # executionBlock = cls.injectPerfCounterKernelWrap(executionBlock, metaInfo)

        executionBlock = super().generateLoopCode(executionBlock, metaInfo, openLoopStatements, ingressDMAStatements,
                                                  egressDMAStatements, closeLoopStatements)
        return executionBlock
