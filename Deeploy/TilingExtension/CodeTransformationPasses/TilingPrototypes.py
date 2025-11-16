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


class ProfilingPrototypeMixIn(ABC):
    _measureCycles = NodeTemplate("""
    ${measurements}[${tileIdxVar}] = getCycles();
    """)

    _measurementArrayDeclaration = NodeTemplate("""
    uint32_t ${measurements}[${totalNumTiles}];
    """)

    _stringDeclaration = NodeTemplate("""
    const static char ${name}[] = "${string}";
    """)

    _printLoopSetup = NodeTemplate("""
    StopTimer();
    for (int ${profileIdxVar} = ((*${tileIdxPtr} > 0) ? ${numTiles}[(*${tileIdxPtr} - 1)] : 0);
        ${profileIdxVar} < ${numTiles}[*${tileIdxPtr}];
        ${profileIdxVar}++){
    """)

    _printCycleDifference = NodeTemplate(r"""
    printf("%s%u] %s%u%s", ${prefixStr}, ${profileIdxVar}, "${flavorStr}", \
    ${measurementsEnd}[${profileIdxVar}] - ${measurementsStart}[${profileIdxVar}], ${suffixStr});
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
            cls._printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Input DMA took ",
                "measurementsStart": f"{nodeName}_ingress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_ingress_dma_wait_end_measurements",
                "profileIdxVar": profileIdxVar,
            })

        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                cls._printCycleDifference, {
                    "prefixStr": f"{nodeName}_prefix",
                    "suffixStr": f"{nodeName}_suffix",
                    "flavorStr": "Kernel took ",
                    "measurementsStart": f"{nodeName}_kernel_start_measurements",
                    "measurementsEnd": f"{nodeName}_kernel_end_measurements",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(
            cls._printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Output DMA took ",
                "measurementsStart": f"{nodeName}_egress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_egress_dma_wait_end_measurements",
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
