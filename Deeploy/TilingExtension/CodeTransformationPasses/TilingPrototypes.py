# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
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


_CodeSegmentType = List[CodeSnippet]

_measureCycles = NodeTemplate("""
${measurements}[${tileIdxVar}] = getCycles();
""")

_measurementArrayDeclaration = NodeTemplate("""
uint32_t ${measurements}[${totalNumTiles}];
""")

_stringDeclaration = NodeTemplate("""
const static char ${name}[] = "${string}";
""")

_measureConditionSetup = NodeTemplate("""
if(${cond}){
""")

_measureConditionEnd = NodeTemplate("""
}
""")

_printLoopSetup = NodeTemplate("""
StopTimer();
for (int ${profileIdxVar} = ${numTiles}[*${tileIdxPtr} -1]; ${profileIdxVar} < ${numTiles}[*${tileIdxPtr}]; ${profileIdxVar}++){
""")

_printCycleDifference = NodeTemplate(r"""
printf("%s%u] %s%u%s", ${prefixStr}, ${profileIdxVar}, "${flavorStr}", \
${measurementsEnd}[${profileIdxVar}] - ${measurementsStart}[${profileIdxVar}], ${suffixStr});
""")

_printLoopTeardown = NodeTemplate("""
}
StartTimer();
""")


class PrototypeTilingMixIn(ABC):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: _CodeSegmentType,
                                     teardownStatements: _CodeSegmentType) -> ExecutionBlock:

        for transaction in reversed(setupStatements):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in teardownStatements:
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock

    @classmethod
    def generateLoopCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                         openLoopStatements: _CodeSegmentType, closeLoopStatements: _CodeSegmentType) -> ExecutionBlock:

        for transaction in reversed(openLoopStatements):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in closeLoopStatements:
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock

    @classmethod
    def generateAllTilingCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                              ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                              ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                              egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                              variableUpdates: _CodeSegmentType, openLoopStatements: _CodeSegmentType,
                              closeLoopStatements: _CodeSegmentType, setupStatements: _CodeSegmentType,
                              teardownStatements: _CodeSegmentType) -> ExecutionBlock:

        if not hasattr(cls, "generateInnerCode"):
            raise Exception("You need to mix in a code gen strategy!")

        newExecutionBlock = cls.generateInnerCode(executionBlock, metaInfo, ingressDMATransferCalls,
                                                  ingressDMAWaitStatements, ingressDMAUpdates, egressDMATransferCalls,
                                                  egressDMAWaitStatements, egressDMAUpdates, variableUpdates)

        newExecutionBlock = cls.generateLoopCode(newExecutionBlock, metaInfo, openLoopStatements, closeLoopStatements)

        newExecutionBlock = cls.generateSetupAndTeardownCode(newExecutionBlock, metaInfo, setupStatements,
                                                             teardownStatements)

        return newExecutionBlock


class TilingCodeGenMixin(ABC):

    @abstractmethod
    def generateInnerCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                          ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                          ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                          egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                          variableUpdates: _CodeSegmentType) -> ExecutionBlock:

        return executionBlock


class ProfilingPrototypeMixIn(ABC):

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
            executionBlock.addLeft(_measurementArrayDeclaration, {
                "measurements": f"{nodeName}_{measurements}_measurements",
                "totalNumTiles": totalNumTiles
            })

        executionBlock.addLeft(_stringDeclaration, {
            "name": f"{nodeName}_prefix",
            "string": f"[{nodeName}][{bufferingStr}][{nodeOps} ops][Tile ",
        })

        executionBlock.addLeft(_stringDeclaration, {
            "name": f"{nodeName}_suffix",
            "string": " cycles \\n",
        })

        return executionBlock

    @classmethod
    def injectPrintCycleDiff(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:

        numTiles = metaInfo.numTiles
        nodeName = metaInfo.nodeName
        tileIdxPtr = metaInfo.tileIdxPtr
        totalNumTiles = metaInfo.totalNumTiles
        profileIdxVar = "PROFILING_I"

        executionBlock.addRight(_printLoopSetup, {
            "numTiles": numTiles,
            "nodeName": nodeName,
            "profileIdxVar": profileIdxVar,
            "tileIdxPtr": tileIdxPtr,
        })

        executionBlock.addRight(
            _printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Input DMA took ",
                "measurementsStart": f"{nodeName}_ingress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_ingress_dma_wait_end_measurements",
                "profileIdxVar": profileIdxVar,
            })

        if metaInfo.kernelLevelTiling:
            executionBlock.addRight(
                _printCycleDifference, {
                    "prefixStr": f"{nodeName}_prefix",
                    "suffixStr": f"{nodeName}_suffix",
                    "flavorStr": "Kernel took ",
                    "measurementsStart": f"{nodeName}_kernel_start_measurements",
                    "measurementsEnd": f"{nodeName}_kernel_end_measurements",
                    "profileIdxVar": profileIdxVar,
                })

        executionBlock.addRight(
            _printCycleDifference, {
                "prefixStr": f"{nodeName}_prefix",
                "suffixStr": f"{nodeName}_suffix",
                "flavorStr": "Output DMA took ",
                "measurementsStart": f"{nodeName}_egress_dma_wait_start_measurements",
                "measurementsEnd": f"{nodeName}_egress_dma_wait_end_measurements",
                "profileIdxVar": profileIdxVar,
            })

        executionBlock.addRight(_printLoopTeardown, {})

        return executionBlock

    @classmethod
    def kernelProfilingWrap(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo) -> ExecutionBlock:
        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        if metaInfo.kernelLevelTiling:
            executionBlock.addLeft(_measureCycles, {
                "measurements": f"{nodeName}_kernel_start_measurements",
                "tileIdxVar": tileIdxVar
            })
            executionBlock.addRight(_measureCycles, {
                "measurements": f"{nodeName}_kernel_end_measurements",
                "tileIdxVar": tileIdxVar
            })

        return executionBlock


class SingleBufferingTilingMixIn(PrototypeTilingMixIn, TilingCodeGenMixin):

    @classmethod
    def generateInnerCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                          ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                          ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                          egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                          variableUpdates: _CodeSegmentType) -> ExecutionBlock:

        # Structure:
        # Update DMA Structs
        # Transfer in tiles (async)
        # Update tile variables
        # Wait for tiles

        # Kernel execution

        # Update DMA Structs
        # Transfer out tiles (async)
        # Wait for out transfers

        for transaction in reversed(ingressDMAUpdates + ingressDMATransferCalls + variableUpdates +
                                    ingressDMAWaitStatements):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in (egressDMAUpdates + egressDMATransferCalls + egressDMAWaitStatements):
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock


class ProfilingSingleBufferingTilingMixIn(SingleBufferingTilingMixIn, ProfilingPrototypeMixIn):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: _CodeSegmentType,
                                     teardownStatements: _CodeSegmentType) -> ExecutionBlock:

        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)

        executionBlock = cls.measurementArrayDeclaration(executionBlock, metaInfo, bufferingStr = "SB")

        executionBlock = cls.injectPrintCycleDiff(executionBlock, metaInfo)

        return executionBlock

    @classmethod
    def generateInnerCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                          ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                          ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                          egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                          variableUpdates: _CodeSegmentType) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        executionBlock = cls.kernelProfilingWrap(executionBlock, metaInfo)

        _ingressDMAWaitStatements = []
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _ingressDMAWaitStatements += ingressDMAWaitStatements
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        _egressDMAWaitStatements = []
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _egressDMAWaitStatements += egressDMAWaitStatements
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        executionBlock = super().generateInnerCode(executionBlock, metaInfo, ingressDMATransferCalls,
                                                   _ingressDMAWaitStatements, ingressDMAUpdates, egressDMATransferCalls,
                                                   _egressDMAWaitStatements, egressDMAUpdates, variableUpdates)

        return executionBlock


class DoubleBufferingTilingMixIn(PrototypeTilingMixIn, TilingCodeGenMixin):

    @classmethod
    def generateInnerCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                          ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                          ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                          egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                          variableUpdates: _CodeSegmentType) -> ExecutionBlock:

        # Structure:

        # Update input DMA Structs
        # Update tile variables
        # Wait for current input tiles
        # Transfer in next input tiles (async)
        # Update output DMA Structs
        # Wait for current output tiles

        # Kernel execution

        # Transfer out tiles (async)

        for transaction in reversed(ingressDMAWaitStatements + ingressDMAUpdates + ingressDMATransferCalls +
                                    variableUpdates + egressDMAWaitStatements + egressDMAUpdates):
            executionBlock.addLeft(transaction.template, transaction.operatorRepresentation)

        for transaction in egressDMATransferCalls:
            executionBlock.addRight(transaction.template, transaction.operatorRepresentation)

        return executionBlock


class ProfilingDoubleBufferingTilingMixIn(DoubleBufferingTilingMixIn, ProfilingPrototypeMixIn):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: _CodeSegmentType,
                                     teardownStatements: _CodeSegmentType) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        totalNumTiles = metaInfo.totalNumTiles

        executionBlock.addLeft(_measureCycles, {
            "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
            "tileIdxVar": 0
        })

        executionBlock = cls.measurementArrayDeclaration(executionBlock, metaInfo, bufferingStr = "DB")

        executionBlock.addRight(_measureCycles, {
            "measurements": f"{nodeName}_egress_dma_wait_start_measurements",
            "tileIdxVar": totalNumTiles - 1
        })
        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)
        executionBlock.addRight(_measureCycles, {
            "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
            "tileIdxVar": totalNumTiles - 1
        })

        executionBlock = cls.injectPrintCycleDiff(executionBlock, metaInfo)

        return executionBlock

    @classmethod
    def generateInnerCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                          ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                          ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                          egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                          variableUpdates: _CodeSegmentType) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        tileIdxVar = metaInfo.tileIdxVar

        executionBlock = cls.kernelProfilingWrap(executionBlock, metaInfo)

        _ingressDMAWaitStatements = []
        _ingressDMAWaitStatements.append(CodeSnippet(_measureConditionSetup, {"cond": f"{tileIdxVar} > 0"}))
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_start_measurements",
                "tileIdxVar": tileIdxVar
            }))
        _ingressDMAWaitStatements.append(CodeSnippet(_measureConditionEnd, {}))
        _ingressDMAWaitStatements += ingressDMAWaitStatements
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_ingress_dma_wait_end_measurements",
                "tileIdxVar": tileIdxVar
            }))

        _egressDMAWaitStatements = []
        _egressDMAWaitStatements.append(CodeSnippet(_measureConditionSetup, {"cond": f"{tileIdxVar} > 0"}))
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_start_measurements",
                "tileIdxVar": f"{tileIdxVar} - 1"
            }))
        _egressDMAWaitStatements += egressDMAWaitStatements
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "measurements": f"{nodeName}_egress_dma_wait_end_measurements",
                "tileIdxVar": f"{tileIdxVar} - 1"
            }))
        _egressDMAWaitStatements.append(CodeSnippet(_measureConditionEnd, {}))

        executionBlock = super().generateInnerCode(executionBlock, metaInfo, ingressDMATransferCalls,
                                                   _ingressDMAWaitStatements, ingressDMAUpdates, egressDMATransferCalls,
                                                   _egressDMAWaitStatements, egressDMAUpdates, variableUpdates)

        return executionBlock
