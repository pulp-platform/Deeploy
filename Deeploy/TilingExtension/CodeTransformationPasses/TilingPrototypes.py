# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from dataclasses import dataclass
from typing import List, Literal

from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NodeTemplate


@dataclass
class TilingMetaInfo:
    nodeName: str
    nodeOps: int
    numTiles: str
    totalNumTiles: int
    tileIdxPtr: str
    tileIdxVar: str
    kernelLevelTiling: bool


class PrototypeTilingMixIn(ABC):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: List[CodeSnippet],
                                     teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        for transaction in reversed(setupStatements):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in teardownStatements:
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock

    @classmethod
    def generateLoopCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                         openLoopStatements: List[CodeSnippet], ingressDMAStatements: List[CodeSnippet],
                         egressDMAStatements: List[CodeSnippet],
                         closeLoopStatements: List[CodeSnippet]) -> ExecutionBlock:

        for transaction in reversed(openLoopStatements + ingressDMAStatements):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in egressDMAStatements + closeLoopStatements:
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock

    @classmethod
    def generateAllTilingCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                              ingressDMAStatements: List[CodeSnippet], egressDMAStatements: List[CodeSnippet],
                              openLoopStatements: List[CodeSnippet], closeLoopStatements: List[CodeSnippet],
                              setupStatements: List[CodeSnippet],
                              teardownStatements: List[CodeSnippet]) -> ExecutionBlock:

        executionBlock = cls.generateLoopCode(executionBlock, metaInfo, openLoopStatements, ingressDMAStatements,
                                              egressDMAStatements, closeLoopStatements)

        executionBlock = cls.generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements, teardownStatements)

        return executionBlock


class PerfCounterProfilingMixIn(ABC):
    """
    MixIn for injecting performance counter profiling code.
    Provides detailed instruction-level statistics using CSR performance counters.
    """

    _perfCounterInit = NodeTemplate("""
    perf_stats_t ${nodeName}_perf_start, ${nodeName}_perf_end, ${nodeName}_perf_total;
    if (pi_core_id() == 0) {
        perf_bench_init();
        perf_bench_start();
        perf_bench_read(&${nodeName}_perf_start);
    }
    """)

    _perfCounterStop = NodeTemplate("""
    if (pi_core_id() == 0) {
        perf_bench_stop();
        perf_bench_read(&${nodeName}_perf_end);
        perf_bench_diff(&${nodeName}_perf_total, &${nodeName}_perf_end, &${nodeName}_perf_start);
        perf_bench_print("${nodeName}", &${nodeName}_perf_total);
    }
    """)

    _perfCounterKernelStart = NodeTemplate("""
    if (pi_core_id() == 0) {
        perf_bench_start();
        perf_bench_read(&${nodeName}_perf_kernel_start);
    }
    """)

    _perfCounterKernelEnd = NodeTemplate("""
    if (pi_core_id() == 0) {
        perf_bench_stop();
        perf_bench_read(&${nodeName}_perf_kernel_end);
        perf_bench_diff(&${nodeName}_perf_kernel_total, &${nodeName}_perf_kernel_end, &${nodeName}_perf_kernel_start);
        perf_bench_print("${nodeName} Kernel", &${nodeName}_perf_kernel_total);
    }
    """)

    _perfCounterKernelDecl = NodeTemplate("""
    perf_stats_t ${nodeName}_perf_kernel_start, ${nodeName}_perf_kernel_end, ${nodeName}_perf_kernel_total;
    """)

    @classmethod
    def injectPerfCounterInit(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:
        """
        Inject performance counter initialization at the beginning of the node execution.
        This should be called in the setup phase.
        """
        nodeName = metaInfo.nodeName

        executionBlock.addLeft(cls._perfCounterInit, {
            "nodeName": nodeName,
        })

        return executionBlock

    @classmethod
    def injectPerfCounterStop(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:
        """
        Inject performance counter stop and print at the end of the node execution.
        This should be called in the teardown phase.
        """
        nodeName = metaInfo.nodeName

        executionBlock.addRight(cls._perfCounterStop, {
            "nodeName": nodeName,
        })

        return executionBlock

    @classmethod
    def injectPerfCounterKernelWrap(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:
        """
        Wrap the kernel execution with performance counter measurements.
        This provides detailed statistics for just the kernel computation (excluding DMA).
        """
        nodeName = metaInfo.nodeName

        if metaInfo.kernelLevelTiling:
            # Add declaration at the beginning
            executionBlock.addLeft(cls._perfCounterKernelDecl, {
                "nodeName": nodeName,
            })

            # Add start measurement before kernel
            executionBlock.addLeft(cls._perfCounterKernelStart, {
                "nodeName": nodeName,
            })

            # Add stop and print after kernel
            executionBlock.addRight(cls._perfCounterKernelEnd, {
                "nodeName": nodeName,
            })

        return executionBlock


class ProfilingPrototypeMixIn(ABC):
    _measureCycles = NodeTemplate("""
    ${measurements}[${tileIdxVar}] = getCycles();
    """)

    # Live, immediately-flushed progress beacon printed *inside* the tile loop.
    # Unlike the cycle measurements (stored in arrays and printed at teardown),
    # this prints synchronously at each phase boundary, so when a tile deadlocks
    # the last beacon on the UART pinpoints exactly which sub-phase hung.
    _liveBeacon = NodeTemplate("""
    if (pi_core_id() == 0) { printf("[TRACE] ${nodeName} tile %u: ${phase}\\r\\n", ${tileIdxVar}); }
    """)

    _measurementArrayDeclaration = NodeTemplate("""
    uint32_t ${measurements}[${totalNumTiles}];
    """)

    _stringDeclaration = NodeTemplate("""
    const static char ${name}[] = "${string}";
    """)

    _printLoopSetup = NodeTemplate("""
    StopTimer();
    printf("===== Profiling ${nodeName} =====\\n");
    for (int ${profileIdxVar} = ((*${tileIdxPtr} > 0) ? ${numTiles}[(*${tileIdxPtr} - 1)] : 0);
        ${profileIdxVar} < ${numTiles}[*${tileIdxPtr}];
        ${profileIdxVar}++){
    """)

    _measurementDeclaration = NodeTemplate("""
    uint32_t ${measurement} = ${measurementsEnd}[${profileIdxVar}] - ${measurementsStart}[${profileIdxVar}];
    """)

    _printCycleDifference = NodeTemplate("""
    printf("%s%u] %s%6u%s", ${prefixStr}, ${profileIdxVar}, "${flavorStr}", \
    ${measurement}, ${suffixStr});
    """)

    _printCycleContribution = NodeTemplate("""
    uint32_t total = ${measurementInput} + ${measurementKernel} + ${measurementOutput};
    uint32_t dma = ${measurementInput} + ${measurementOutput};
    float overhead_percentage = (total == 0) ? 0 : dma * 100.0f / total;
    float kernel_percentage = (total == 0) ? 0 : ${measurementKernel} * 100.0f / total;
    printf("%s%u] Total      :%6u cycles (%2.1f%% Kernel + %2.1f%% Overhead, %u + %u)\\n", ${prefixStr}, ${profileIdxVar}, total, kernel_percentage, overhead_percentage    , ${measurementKernel}, dma);
    """)

    _printLoopTeardown = NodeTemplate("""
    }
    StartTimer();
    """)

    _measureConditionSetup = NodeTemplate("""
    if(${cond}){
    """)

    _measureConditionEnd = NodeTemplate("""
    }
    """)

    @classmethod
    def measurementArrayDeclaration(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                    bufferingStr: Literal["SB", "DB"]) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        numTiles = metaInfo.numTiles
        totalNumTiles = metaInfo.totalNumTiles
        nodeOps = metaInfo.nodeOps

        measurementsList = [
            "ingress_dma_wait_start", "ingress_dma_wait_end", "egress_dma_wait_start", "egress_dma_wait_end"
        ]

        if metaInfo.kernelLevelTiling:
            measurementsList = ["kernel_start", "kernel_end"] + measurementsList

        for measurements in measurementsList:
            executionBlock.addLeft(cls._measurementArrayDeclaration, {
                "measurements": f"{nodeName}_{measurements}_measurements",
                "totalNumTiles": totalNumTiles
            })

        executionBlock.addLeft(cls._stringDeclaration, {
            "name": f"{nodeName}_prefix",
            "string": f"[{nodeName}][{bufferingStr}][{nodeOps} ops][Tile ",
        })

        executionBlock.addLeft(cls._stringDeclaration, {
            "name": f"{nodeName}_suffix",
            "string": " cycles \\n",
        })

        return executionBlock

    @classmethod
    def injectPrintCycleDiff(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:

        numTiles = metaInfo.numTiles
        nodeName = metaInfo.nodeName
        tileIdxPtr = metaInfo.tileIdxPtr
        profileIdxVar = "PROFILING_I"

        executionBlock.addRight(cls._printLoopSetup, {
            "numTiles": numTiles,
            "nodeName": nodeName,
            "profileIdxVar": profileIdxVar,
            "tileIdxPtr": tileIdxPtr,
        })

        executionBlock.addRight(
            cls._measurementDeclaration, {
                "measurement": f"{nodeName}_ingress_dma_wait_measurement",
                "measurementsStart": f"{nodeName}_ingress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_ingress_dma_wait_end_measurements",
                "profileIdxVar": profileIdxVar,
            })

        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                cls._measurementDeclaration, {
                    "measurement": f"{nodeName}_kernel_measurement",
                    "measurementsStart": f"{nodeName}_kernel_start_measurements",
                    "measurementsEnd": f"{nodeName}_kernel_end_measurements",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(
            cls._measurementDeclaration, {
                "measurement": f"{nodeName}_egress_dma_wait_measurement",
                "measurementsStart": f"{nodeName}_egress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_egress_dma_wait_end_measurements",
                "profileIdxVar": profileIdxVar,
            })

        executionBlock.addRight(
            cls._printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Pre-Kernel :",
                "measurement": f"{nodeName}_ingress_dma_wait_measurement",
                "profileIdxVar": profileIdxVar,
            })

        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                cls._printCycleDifference, {
                    "prefixStr": f"{nodeName}_prefix",
                    "suffixStr": f"{nodeName}_suffix",
                    "flavorStr": "Kernel     :",
                    "measurement": f"{nodeName}_kernel_measurement",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(
            cls._printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Post-Kernel:",
                "measurement": f"{nodeName}_egress_dma_wait_measurement",
                "profileIdxVar": profileIdxVar,
            })

        # Total Time: Input + Kernel + Output
        # Overhead: (Input + Output) / Total
        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                cls._printCycleContribution, {
                    "prefixStr": f"{nodeName}_prefix",
                    "measurementInput": f"{nodeName}_ingress_dma_wait_measurement",
                    "measurementKernel": f"{nodeName}_kernel_measurement",
                    "measurementOutput": f"{nodeName}_egress_dma_wait_measurement",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(cls._printLoopTeardown, {})

        return executionBlock

    @classmethod
    def kernelProfilingWrap(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:
        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        if metaInfo.kernelLevelTiling:
            executionBlock.addLeft(cls._measureCycles, {
                "measurements": f"{nodeName}_kernel_start_measurements",
                "tileIdxVar": tileIdxVar
            })
            executionBlock.addRight(cls._measureCycles, {
                "measurements": f"{nodeName}_kernel_end_measurements",
                "tileIdxVar": tileIdxVar
            })

        return executionBlock
