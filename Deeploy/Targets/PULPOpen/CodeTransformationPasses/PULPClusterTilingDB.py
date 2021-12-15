# ----------------------------------------------------------------------
#
# File: PULPClusterTilingDB.py
#
# Last edited: 25.10.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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

import copy
from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import CodeSnippet, ExecutionBlock, NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterTilingSB import PULPClusterTilingSB, _DMAUpdate
from Deeploy.Targets.PULPOpen.DataTypes import PULPStructDataTypes
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import DoubleBufferingTilingMixIn, \
    ProfilingDoubleBufferingTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
if (${tileNum} < ${numTiles}[*${tileIdxPtr}+1]){
dory_dma_memcpy_mindims_async(&${stateReference});
}

""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
if((${tileNum}) % 2 == 0){
dory_dma_memcpy_mindims_async(&${stateReference});
} else {
dory_dma_memcpy_mindims_async(&${_stateReference});
}
""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
if((${tileNum}) > 1){
if((${tileNum}) % 2 == 0){
dory_dma_barrier(&${stateReference});
} else {
dory_dma_barrier(&${_stateReference});
}
}

""")

_finalBlockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
dory_dma_barrier(&${stateReference});
dory_dma_barrier(&${_stateReference});
""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}, ${_stateReference}
${stateReference}.ext = (((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}]);
${stateReference}.mchan_cmd = ${mchanCmdPtr}[${tileNum}];
${stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];
${stateReference}.loc = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}-1]);
""")

_outUpdateDMATransferStructTemplate = NodeTemplate("""

if ((${tileNum}) % 2 == 0){
// UPDATE DMA STRUCT ${stateReference}
${stateReference}.ext = ((char*)${extPtr} + ${extOffsetPtr}[${tileNum}]);
${stateReference}.mchan_cmd = ${mchanCmdPtr}[${tileNum}];
${stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];
${stateReference}.loc = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
} else {
${_stateReference}.ext = ((char*)${extPtr} + ${extOffsetPtr}[${tileNum}]);
${_stateReference}.mchan_cmd = ${mchanCmdPtr}[${tileNum}];
${_stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${_stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${_stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];
${_stateReference}.loc = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
}
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);

""")


class PULPClusterTilingDB(PULPClusterTilingSB):

    _blockTileOutTemplate = _blockTileOutTemplate
    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate
    _moveTileOutTemplate = _moveTileOutTemplate
    _moveTileInTemplate = _moveTileInTemplate

    def _hoistDMAUpdates(self, ctxt: NetworkContext, tensorName: str, updateList: List[_DMAUpdate],
                         operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:
        nodeName = operatorRepresentation['nodeName']

        operatorRepresentation = operatorRepresentation.copy()

        dmaName = self._DMAStructName(tensorName, nodeName)
        # operatorRepresentation['stateReference'] = dmaName
        # operatorRepresentation['tileNum'] = "TILING_I"
        operatorRepresentation['locPtr'] = ctxt.lookup(operatorRepresentation[tensorName]).name
        operatorRepresentation['baseLocPtr'] = ctxt.hoistReference(operatorRepresentation['locPtr'],
                                                                   operatorRepresentation['locPtr'] + "_ref")
        operatorRepresentation['_stateReference'] = self._DMAStructName(tensorName, nodeName) + "_1"
        ctxt.lookup(operatorRepresentation['baseLocPtr'])._memoryLevel = self.targetMemLevel

        namePrefix = self.prefix + f"{nodeName}_{tensorName}"

        ctxt, operatorRepresentation = super()._hoistDMAUpdates(ctxt, tensorName, updateList, operatorRepresentation)

        locOffsetList = []
        locBaseOffset = updateList[0].locOffset
        for update in updateList:
            locOffsetList.append(int(update.locOffset) - locBaseOffset)

        name = namePrefix + "_locOffset"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], locOffsetList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(ctxt, cb, operatorRepresentation, nodeName,
                                                                       'locOffsetPtr')

        return ctxt, operatorRepresentation

    def _generateEgressPointerUpdates(
            self, nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, List[CodeSnippet]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = self._generatePointerUpdates(ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
                                                  nodeMemoryConstraint, tilingSchedule)

        for key, updateList in updateDict.items():

            newCtxt, newNodeRep = self._hoistDMAUpdates(newCtxt, key, updateList, operatorRepresentation)
            updates.append(CodeSnippet(_outUpdateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateEgressDMACode(
            self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        egressDMATransferCalls = []
        egressDMAWaitStatements = []

        exportLoadStep = tilingSchedule.outputLoadSchedule[0]
        for key, rectangle in exportLoadStep.items():
            externalPtr = ctxt.lookup(ctxt.lookup(operatorRepresentation[key])._referenceName)
            internalPtr = ctxt.lookup(operatorRepresentation[key])

            tensorName = key
            nodeName = operatorRepresentation['nodeName']
            dmaName = self._DMAStructName(tensorName, nodeName)

            finalMemoryLevel = TilingCodeGeneration.isFinalMemoryLevel(nodeMemoryConstraint, internalPtr)
            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL1", internalPtr.name, externalPtr.name,
                                           finalMemoryLevel)
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.DMA_copy)
            ctxt.lookup(dmaName)._users += [operatorRepresentation['nodeName']]

            tensorName = key + "_1"
            nodeName = operatorRepresentation['nodeName']
            _dmaName = self._DMAStructName(tensorName, nodeName)

            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL1", internalPtr.name, externalPtr.name,
                                           finalMemoryLevel)
            _ = ctxt.hoistStruct(struct, _dmaName, PULPStructDataTypes.DMA_copy)
            ctxt.lookup(_dmaName)._users += [operatorRepresentation['nodeName']]

            egressDMATransferCalls.append(
                CodeSnippet(
                    self._moveTileOutTemplate, {
                        'innerTilePtr': str(internalPtr._instance),
                        "outerTilePtr": str(externalPtr._instance),
                        "stateReference": dmaName,
                        "_stateReference": _dmaName
                    }))

            egressDMAWaitStatements.append(
                CodeSnippet(
                    self._blockTileOutTemplate, {
                        'innerTilePtr': str(internalPtr._instance),
                        "outerTilePtr": str(externalPtr._instance),
                        "stateReference": dmaName,
                        "_stateReference": _dmaName
                    }))

        return egressDMATransferCalls, egressDMAWaitStatements

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        tileIdxPtr = self._hoistTileIdxPtr(ctxt, operatorRepresentation)

        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateIngressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, operatorRepresentation)

        egressDMATransferCalls, egressDMAWaitStatements = self._generateEgressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, operatorRepresentation)

        ctxt, ingressDMAUpdates = self._generateIngressPointerUpdates(nodeMemoryConstraint, tilingSchedule, ctxt,
                                                                      operatorRepresentation)
        ctxt, egressDMAUpdates = self._generateEgressPointerUpdates(nodeMemoryConstraint, tilingSchedule, ctxt,
                                                                    operatorRepresentation)

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt,
                                                        operatorRepresentation)

        for transaction in ingressDMATransferCalls:
            _operatorRepresentation = transaction.operatorRepresentation
            _operatorRepresentation["tileNum"] = "TILING_I+1"
            _operatorRepresentation["numTiles"] = operatorRepresentation['numTiles']
            _operatorRepresentation["tileIdxPtr"] = tileIdxPtr

        for transaction in ingressDMAUpdates:
            _operatorRepresentation = transaction.operatorRepresentation
            _operatorRepresentation["tileNum"] = "TILING_I+1"

        for transaction in egressDMATransferCalls:
            _operatorRepresentation = transaction.operatorRepresentation
            _operatorRepresentation["tileNum"] = "TILING_I"

        for transaction in egressDMAWaitStatements:
            _operatorRepresentation = transaction.operatorRepresentation
            _operatorRepresentation['tileNum'] = "TILING_I"

        for transaction in egressDMAUpdates:
            _operatorRepresentation = transaction.operatorRepresentation
            _operatorRepresentation["tileNum"] = "TILING_I"

        for transaction in variableUpdates:
            _operatorRepresentation = transaction.operatorRepresentation
            _operatorRepresentation["tileNum"] = "TILING_I"

        openLoopStatement = [
            CodeSnippet(self._openTileLoopTemplate, {
                "numTiles": operatorRepresentation["numTiles"],
                "tileIdxPtr": tileIdxPtr
            })
        ]

        closeLoopStatement = [
            CodeSnippet(self._closeTileLoopTemplate, {
                "numTiles": operatorRepresentation["numTiles"],
                "tileIdxPtr": tileIdxPtr
            })
        ]

        setupStatements = []
        teardownStatements = []

        teardownStatements += [
            CodeSnippet(self._releaseDMATemplate,
                        {"stateReference": ingressDMAUpdates[0].operatorRepresentation["stateReference"]})
        ]

        setupStatements += [CodeSnippet(self._initDMATemplate, {"channelName": "dma_channel"})]
        setupStatements += [
            CodeSnippet(self._setDMAChannelTemplate, {
                **transaction.operatorRepresentation, "channelName": "dma_channel"
            }) for transaction in ingressDMAUpdates
        ]

        for transaction in egressDMAUpdates:
            _operatorRepresentation = transaction.operatorRepresentation.copy()
            _operatorRepresentation["channelName"] = "dma_channel"
            setupStatements.append(CodeSnippet(self._setDMAChannelTemplate, _operatorRepresentation.copy()))
            _operatorRepresentation["channelName"] = "dma_channel"
            _operatorRepresentation["stateReference"] = _operatorRepresentation["_stateReference"]
            setupStatements.append(CodeSnippet(self._setDMAChannelTemplate, _operatorRepresentation.copy()))

        for transaction in ingressDMATransferCalls:
            _operatorRepresentation = transaction.operatorRepresentation.copy()
            _operatorRepresentation["tileNum"] = 0
            _operatorRepresentation["numTiles"] = operatorRepresentation['numTiles']
            _operatorRepresentation["tileIdxPtr"] = tileIdxPtr
            setupStatements.append(CodeSnippet(transaction.template, _operatorRepresentation))

        for transaction in egressDMAWaitStatements:
            _operatorRepresentation = transaction.operatorRepresentation.copy()
            _operatorRepresentation['tileNum'] = ctxt.lookup(operatorRepresentation["numTiles"]).values[-1]
            teardownStatements.append(CodeSnippet(_finalBlockTileOutTemplate, _operatorRepresentation))

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + "_L2",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxVar = "TILING_I")

        newExecutionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDMATransferCalls,
                                                       ingressDMAWaitStatements[-1:], ingressDMAUpdates,
                                                       egressDMATransferCalls, egressDMAWaitStatements[-1:],
                                                       egressDMAUpdates, variableUpdates, openLoopStatement,
                                                       closeLoopStatement, setupStatements, teardownStatements)

        return ctxt, newExecutionBlock, True

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
            if not len(offsetList) == 2:
                return ctxt, executionBlock, False

        allNumTiles = [len(schedule.outputLoadSchedule) for schedule in tilingSchedules]
        operatorRepresentation["numTiles"] = self._hoistNumTiles(ctxt, operatorRepresentation['nodeName'],
                                                                 tilingSchedules)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)


class PULPClusterTilingGenerationDB(PULPClusterTilingDB, DoubleBufferingTilingMixIn):
    pass


class ProfilingPULPClusterTilingGenerationDB(PULPClusterTilingDB, ProfilingDoubleBufferingTilingMixIn):
    pass
