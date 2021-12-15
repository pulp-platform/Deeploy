# ----------------------------------------------------------------------
#
# File: TilingPrototypes.py
#
# Last edited: 17.04.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NodeTemplate


@dataclass
class TilingMetaInfo:
    nodeName: str
    nodeOps: int
    numTiles: int
    tileIdxVar: str


_CodeSegmentType = List[CodeSnippet]

_measureCycles = NodeTemplate("""
${nodeName}_${measurementName}_measurements[${tileIdx}] = getCycles();
""")

_measurementArrayDeclaration = NodeTemplate("""
uint32_t ${nodeName}_${measurementName}_measurements[${numTiles}];
""")

_printPrefixAndSufixDeclaration = NodeTemplate("""
char ${nodeName}_prefix[] = "[${nodeName}][${buffering}][${nodeOps} ops][Tile ";
char ${nodeName}_suffix[] = " cycles \\n";
""")

_measureConditionSetup = NodeTemplate("""
if(${cond}){
""")

_measureConditionEnd = NodeTemplate("""
}
""")

_printLoopSetup = NodeTemplate("""
StopTimer();
for (int printLoopIdx = 0; printLoopIdx < ${numTiles}; printLoopIdx++){
""")

_printCycleDifference = NodeTemplate(r"""
printf("%s%u] %s%u%s", ${nodeName}_prefix,${tileIdx},"${flavorStr}", \
${nodeName}_${endMeasurementName}_measurements[${tileIdx}] - ${nodeName}_${startMeasurementName}_measurements[${tileIdx}],${nodeName}_suffix);
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


class ProfilingSingleBufferingTilingMixIn(SingleBufferingTilingMixIn):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: _CodeSegmentType,
                                     teardownStatements: _CodeSegmentType) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        nodeOps = metaInfo.nodeOps
        numTiles = metaInfo.numTiles

        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)

        for measurementName in [
                "kernel_start", "kernel_end", "ingress_dma_wait_start", "ingress_dma_wait_end", "egress_dma_wait_start",
                "egress_dma_wait_end"
        ]:
            executionBlock.addLeft(_measurementArrayDeclaration, {
                "nodeName": nodeName,
                "measurementName": measurementName,
                "numTiles": numTiles
            })

        executionBlock.addLeft(_printPrefixAndSufixDeclaration, {
            "nodeName": nodeName,
            "nodeOps": nodeOps,
            "buffering": "SB"
        })

        executionBlock.addRight(_printLoopSetup, {"numTiles": numTiles})

        executionBlock.addRight(
            _printCycleDifference, {
                "nodeName": nodeName,
                "flavorStr": "Input DMA took ",
                "startMeasurementName": "ingress_dma_wait_start",
                "endMeasurementName": "ingress_dma_wait_end",
                "tileIdx": "printLoopIdx"
            })
        executionBlock.addRight(
            _printCycleDifference, {
                "nodeName": nodeName,
                "flavorStr": "Kernel took ",
                "startMeasurementName": "kernel_start",
                "endMeasurementName": "kernel_end",
                "tileIdx": "printLoopIdx"
            })
        executionBlock.addRight(
            _printCycleDifference, {
                "nodeName": nodeName,
                "flavorStr": "Output DMA took ",
                "startMeasurementName": "egress_dma_wait_start",
                "endMeasurementName": "egress_dma_wait_end",
                "tileIdx": "printLoopIdx"
            })

        executionBlock.addRight(_printLoopTeardown, {})

        return executionBlock

    @classmethod
    def generateInnerCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                          ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                          ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                          egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                          variableUpdates: _CodeSegmentType) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        numTiles = metaInfo.numTiles
        tileIdxVar = metaInfo.tileIdxVar

        executionBlock.addLeft(_measureCycles, {
            "nodeName": nodeName,
            "measurementName": "kernel_start",
            "tileIdx": tileIdxVar
        })
        executionBlock.addRight(_measureCycles, {
            "nodeName": nodeName,
            "measurementName": "kernel_end",
            "tileIdx": tileIdxVar
        })

        _ingressDMAWaitStatements = []
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "ingress_dma_wait_start",
                "tileIdx": tileIdxVar
            }))
        _ingressDMAWaitStatements += ingressDMAWaitStatements
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "ingress_dma_wait_end",
                "tileIdx": tileIdxVar
            }))

        _egressDMAWaitStatements = []
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "egress_dma_wait_start",
                "tileIdx": tileIdxVar
            }))
        _egressDMAWaitStatements += egressDMAWaitStatements
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "egress_dma_wait_end",
                "tileIdx": tileIdxVar
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


class ProfilingDoubleBufferingTilingMixIn(DoubleBufferingTilingMixIn):

    @classmethod
    def generateSetupAndTeardownCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                                     setupStatements: _CodeSegmentType,
                                     teardownStatements: _CodeSegmentType) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        nodeOps = metaInfo.nodeOps
        numTiles = metaInfo.numTiles

        executionBlock.addLeft(_measureCycles, {
            "nodeName": nodeName,
            "measurementName": "ingress_dma_wait_start",
            "tileIdx": 0
        })

        for measurementName in [
                "kernel_start", "kernel_end", "ingress_dma_wait_start", "ingress_dma_wait_end", "egress_dma_wait_start",
                "egress_dma_wait_end"
        ]:
            executionBlock.addLeft(_measurementArrayDeclaration, {
                "nodeName": nodeName,
                "measurementName": measurementName,
                "numTiles": numTiles
            })

        executionBlock.addLeft(_printPrefixAndSufixDeclaration, {
            "nodeName": nodeName,
            "nodeOps": nodeOps,
            "buffering": "DB"
        })

        executionBlock.addRight(_measureCycles, {
            "nodeName": nodeName,
            "measurementName": "egress_dma_wait_start",
            "tileIdx": numTiles - 1
        })
        executionBlock = super().generateSetupAndTeardownCode(executionBlock, metaInfo, setupStatements,
                                                              teardownStatements)
        executionBlock.addRight(_measureCycles, {
            "nodeName": nodeName,
            "measurementName": "egress_dma_wait_end",
            "tileIdx": numTiles - 1
        })

        executionBlock.addRight(_printLoopSetup, {"numTiles": numTiles})

        executionBlock.addRight(
            _printCycleDifference, {
                "nodeName": nodeName,
                "flavorStr": "Input DMA took ",
                "startMeasurementName": "ingress_dma_wait_start",
                "endMeasurementName": "ingress_dma_wait_end",
                "tileIdx": "printLoopIdx"
            })
        executionBlock.addRight(
            _printCycleDifference, {
                "nodeName": nodeName,
                "flavorStr": "Kernel took ",
                "startMeasurementName": "kernel_start",
                "endMeasurementName": "kernel_end",
                "tileIdx": "printLoopIdx"
            })
        executionBlock.addRight(
            _printCycleDifference, {
                "nodeName": nodeName,
                "flavorStr": "Output DMA took ",
                "startMeasurementName": "egress_dma_wait_start",
                "endMeasurementName": "egress_dma_wait_end",
                "tileIdx": "printLoopIdx"
            })

        executionBlock.addRight(_printLoopTeardown, {})

        return executionBlock

    @classmethod
    def generateInnerCode(cls, executionBlock: ExecutionBlock, metaInfo: TilingMetaInfo,
                          ingressDMATransferCalls: _CodeSegmentType, ingressDMAWaitStatements: _CodeSegmentType,
                          ingressDMAUpdates: _CodeSegmentType, egressDMATransferCalls: _CodeSegmentType,
                          egressDMAWaitStatements: _CodeSegmentType, egressDMAUpdates: _CodeSegmentType,
                          variableUpdates: _CodeSegmentType) -> ExecutionBlock:

        nodeName = metaInfo.nodeName
        numTiles = metaInfo.numTiles
        tileIdxVar = metaInfo.tileIdxVar

        executionBlock.addLeft(_measureCycles, {
            "nodeName": nodeName,
            "measurementName": "kernel_start",
            "tileIdx": tileIdxVar
        })
        executionBlock.addRight(_measureCycles, {
            "nodeName": nodeName,
            "measurementName": "kernel_end",
            "tileIdx": tileIdxVar
        })

        _ingressDMAWaitStatements = []
        _ingressDMAWaitStatements.append(CodeSnippet(_measureConditionSetup, {"cond": f"{tileIdxVar} > 0"}))
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "ingress_dma_wait_start",
                "tileIdx": tileIdxVar
            }))
        _ingressDMAWaitStatements.append(CodeSnippet(_measureConditionEnd, {}))
        _ingressDMAWaitStatements += ingressDMAWaitStatements
        _ingressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "ingress_dma_wait_end",
                "tileIdx": tileIdxVar
            }))

        _egressDMAWaitStatements = []
        _egressDMAWaitStatements.append(CodeSnippet(_measureConditionSetup, {"cond": f"{tileIdxVar} > 0"}))
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "egress_dma_wait_start",
                "tileIdx": f"{tileIdxVar} - 1"
            }))
        _egressDMAWaitStatements += egressDMAWaitStatements
        _egressDMAWaitStatements.append(
            CodeSnippet(_measureCycles, {
                "nodeName": nodeName,
                "measurementName": "egress_dma_wait_end",
                "tileIdx": f"{tileIdxVar} - 1"
            }))
        _egressDMAWaitStatements.append(CodeSnippet(_measureConditionEnd, {}))

        executionBlock = super().generateInnerCode(executionBlock, metaInfo, ingressDMATransferCalls,
                                                   _ingressDMAWaitStatements, ingressDMAUpdates, egressDMATransferCalls,
                                                   _egressDMAWaitStatements, egressDMAUpdates, variableUpdates)

        return executionBlock
