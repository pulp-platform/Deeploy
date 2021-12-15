# ----------------------------------------------------------------------
#
# File: PULPClusterTiling.py
#
# Last edited: 17.10.2023
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
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPL3TilingSB import PULPL3TilingSB, _DMAUpdate
from Deeploy.Targets.PULPOpen.DataTypes import PULPStructDataTypes
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import DoubleBufferingTilingMixIn, \
    ProfilingDoubleBufferingTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule, VariableReplacementScheme

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
if (${tileNum} < ${numTiles}[*${tileIdxPtr}+1]){
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});
}

""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
if((${tileNum}) % 2 == 0){
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});
} else {
pi_cl_ram_copy_2d(get_ram_ptr(), ${_stateReference}.pi_ram_addr, ${_stateReference}.addr, ${_stateReference}.size, ${_stateReference}.stride, ${_stateReference}.length, ${_stateReference}.ext2loc, &${_stateReference});
}

""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
if((${tileNum}) > 1){
if((${tileNum}) % 2 == 0){
pi_cl_ram_copy_wait(&${stateReference});
} else {
pi_cl_ram_copy_wait(&${_stateReference});
}
}

""")

_finalBlockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
pi_cl_ram_copy_wait(&${stateReference});
% if numTiles > 1:
pi_cl_ram_copy_wait(&${_stateReference});
% endif
""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}
${stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.size = ${length1dPtr}[${tileNum}];
${stateReference}.length = ${number1dPtr}[${tileNum}];
${stateReference}.addr = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}-1]);

""")

_outUpdateDMATransferStructTemplate = NodeTemplate("""

if ((${tileNum}) % 2 == 0){
// UPDATE DMA STRUCT ${stateReference}
${stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.size = ${length1dPtr}[${tileNum}];
${stateReference}.length = ${number1dPtr}[${tileNum}];
${stateReference}.addr = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
} else {
${_stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${_stateReference}.size = ${length1dPtr}[${tileNum}];
${_stateReference}.length = ${number1dPtr}[${tileNum}];
${_stateReference}.addr = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);
}
${locPtr} = (((char*)${baseLocPtr}) + ${locOffsetPtr}[${tileNum}]);

""")


class PULPL3TilingDB(PULPL3TilingSB):

    _prefix = "TILING_REPLACED_"
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
            self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, List[CodeSnippet]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = self._generatePointerUpdates(ctxt, operatorRepresentation, tilingSchedule.outputLoadSchedule,
                                                  tilingSchedule)

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

            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL2", internalPtr.name, externalPtr.name)
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.pi_cl_ram_req_t)
            ctxt.lookup(dmaName)._users += [operatorRepresentation['nodeName']]

            tensorName = key + "_1"
            nodeName = operatorRepresentation['nodeName']
            _dmaName = self._DMAStructName(tensorName, nodeName)

            struct = self._rectToDMAStruct(ctxt, rectangle, "FromL2", internalPtr.name, externalPtr.name)
            _ = ctxt.hoistStruct(struct, _dmaName, PULPStructDataTypes.pi_cl_ram_req_t)
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
            tilingSchedule, ctxt, operatorRepresentation)

        egressDMATransferCalls, egressDMAWaitStatements = self._generateEgressDMACode(
            tilingSchedule, nodeMemoryConstraint, ctxt, operatorRepresentation)

        ctxt, ingressDMAUpdates = self._generateIngressPointerUpdates(tilingSchedule, ctxt, operatorRepresentation)
        ctxt, egressDMAUpdates = self._generateEgressPointerUpdates(tilingSchedule, ctxt, operatorRepresentation)

        variableUpdates = []

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

        for transaction in ingressDMATransferCalls:
            _operatorRepresentation = transaction.operatorRepresentation.copy()
            _operatorRepresentation["tileNum"] = 0
            _operatorRepresentation["numTiles"] = operatorRepresentation['numTiles']
            _operatorRepresentation["tileIdxPtr"] = tileIdxPtr
            setupStatements.append(CodeSnippet(transaction.template, _operatorRepresentation))

        for transaction in egressDMAWaitStatements:
            _operatorRepresentation = transaction.operatorRepresentation.copy()
            _operatorRepresentation['tileNum'] = ctxt.lookup(operatorRepresentation["numTiles"]).values[-1]
            _operatorRepresentation['numTiles'] = len(tilingSchedule.outputLoadSchedule)
            teardownStatements.append(CodeSnippet(_finalBlockTileOutTemplate, _operatorRepresentation))

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + "_L3",
                                  nodeOps = operatorRepresentation['nodeOps'],
                                  numTiles = len(tilingSchedule.outputLoadSchedule),
                                  tileIdxVar = "TILING_I")

        newExecutionBlock = self.generateAllTilingCode(executionBlock, metaInfo, ingressDMATransferCalls,
                                                       ingressDMAWaitStatements, ingressDMAUpdates,
                                                       egressDMATransferCalls, egressDMAWaitStatements,
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


class PULPL3TilingGenerationDB(PULPL3TilingDB, DoubleBufferingTilingMixIn):
    pass


class ProfilingPULPL3TilingGenerationDB(PULPL3TilingDB, ProfilingDoubleBufferingTilingMixIn):
    pass
