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
from collections import namedtuple
from typing import Dict, List, Literal, Optional, Tuple, Type

import numpy as np

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import Immediate, PointerClass
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    _invertPermutation, _permuteList
from Deeploy.DeeployTypes import CodeSnippet, ConstantBuffer, ExecutionBlock, NetworkContext, NodeTemplate, \
    OperatorRepresentation
from Deeploy.Targets.PULPOpen.CodeTransformationPasses import AutoTransposeUtils
from Deeploy.Targets.PULPOpen.DataTypes import PULPStructDataTypes
from Deeploy.TilingExtension.CodeTransformationPasses.TilingCodeGeneration import TilingCodeGeneration
from Deeploy.TilingExtension.CodeTransformationPasses.TilingPrototypes import ProfilingSingleBufferingTilingMixIn, \
    SingleBufferingTilingMixIn, TilingMetaInfo
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, TilingSchedule, VariableReplacementScheme, \
    calculateRectangleOffset, minimizeRectangleDims

_openTileLoopTemplate = NodeTemplate("""

// TILING LOOP
for (int TILING_I=${numTiles}[*${tileIdxPtr}]; TILING_I<${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_closeTileLoopTemplate = NodeTemplate("""

// CLOSE TILING LOOP
}
*${tileIdxPtr} += 1;

""")

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
dory_dma_memcpy_mindims_async(&${stateReference});

""")

_iteratedMoveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
// ITERATED

<%
_extStrides = [stride * stateStruct.value['length_1d_copy'].value for stride in remainderStrides]
_locStride = f"{stateReference}.length_1d_copy  * {stateReference}.number_of_1d_copies  *  {stateReference}.number_of_2d_copies"

stateStruct.value['ext'] = str(stateReference) + ".ext"
stateStruct.value['loc'] = str(stateReference) + ".loc"
stateStruct.value['tid'] = str(stateReference) + ".tid"
stateStruct.value['stride_2d'] = str(stateReference) + ".stride_2d"
stateStruct.value['stride_1d'] = str(stateReference) + ".stride_1d"
stateStruct.value['number_of_2d_copies'] = str(stateReference) + ".number_of_2d_copies"
stateStruct.value['number_of_1d_copies'] = str(stateReference) + ".number_of_1d_copies"
stateStruct.value['length_1d_copy'] = str(stateReference) + ".length_1d_copy"
%>

int8_t * bu_${stateReference}_loc = ${stateReference}.loc;
int8_t * bu_${stateReference}_ext = ${stateReference}.ext;

% for idx, dimLen in enumerate(dimLens):
uint16_t ${nodeName}_${tensorName}_dimLen_${idx} = ${dimLen}[${tileNum}];
for(int i_${idx} = 0; i_${idx} < ${nodeName}_${tensorName}_dimLen_${idx}; i_${idx}++){
%endfor
${stateStruct.typeName} trans_${stateReference} = (${stateStruct.typeName}) ${str(stateStruct)};
dory_dma_memcpy_mindims_async(&trans_${stateReference});
${stateStruct.value['loc']} = (((int8_t*) ${stateStruct.value['loc']}) + ${_locStride});
% for idx, _ in enumerate(dimLens):
${stateStruct.value['ext']} = (((int8_t*) ${stateStruct.value['ext']}) + (${_extStrides[idx]}));
}
${stateStruct.value['ext']} = (((int8_t*) ${stateStruct.value['ext']}) - ${nodeName}_${tensorName}_dimLen_${len(dimLens) -1 - idx} * ${_extStrides[idx]});
%endfor

${stateStruct.value['loc']} = bu_${stateReference}_loc;
${stateStruct.value['ext']} = bu_${stateReference}_ext;

""")

_blockTileInTemplate = NodeTemplate("""

// BLOCKING IMPORT TILE ${innerTilePtr}
dory_dma_barrier(&${stateReference});

""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
dory_dma_memcpy_mindims_async(&${stateReference});

""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
dory_dma_barrier(&${stateReference});

""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}
${stateReference}.ext = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.length_1d_copy = ${length1dPtr}[${tileNum}];
${stateReference}.number_of_1d_copies = ${number1dPtr}[${tileNum}];
${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];

${stateReference}.stride_1d = ${stride1dPtr}[${tileNum}];
${stateReference}.stride_2d = ${stride2dPtr}[${tileNum}];

${stateReference}.mchan_cmd = ${mchanCmdPtr}[${tileNum}];
""")

_updateReferenceTemplate = NodeTemplate("""

// UPDATE VARIABLE ${reference}
*${reference} = ${baseReference}[${tileNum}];
""")

_initDMATemplate = NodeTemplate("""
int32_t ${channelName} = dory_dma_allocate();
""")

_setDMAChannelTemplate = NodeTemplate("""
${stateReference}.tid = ${channelName};
""")

_releaseDMATemplate = NodeTemplate("""
dory_dma_free(&${stateReference});
""")

# ADD NUM TRANSFERS VARIABLE

_DMAUpdate = namedtuple(
    "_DMAUpdate",
    "extOffset locOffset length_1d_copy number_of_1d_copies number_of_2d_copies stride_1d stride_2d mchan_cmd")


class PULPClusterTilingSB(TilingCodeGeneration):

    _prefix = "TILING_REPLACED_"

    _openTileLoopTemplate = _openTileLoopTemplate
    _closeTileLoopTemplate = _closeTileLoopTemplate

    _moveTileInTemplate = _moveTileInTemplate
    _iteratedMoveTileInTemplate = _iteratedMoveTileInTemplate
    _blockTileInTemplate = _blockTileInTemplate

    _moveTileOutTemplate = _moveTileOutTemplate
    _blockTileOutTemplate = _blockTileOutTemplate

    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate
    _updateReferenceTemplate = _updateReferenceTemplate

    _initDMATemplate = _initDMATemplate
    _setDMAChannelTemplate = _setDMAChannelTemplate
    _releaseDMATemplate = _releaseDMATemplate

    @property
    def prefix(self):
        return self._prefix + self.targetMemLevel + "_"

    def _DMAStructName(self, tensorName: str, nodeName: str) -> str:
        return f"{self.prefix}_DMA_{nodeName}_{tensorName}"

    @classmethod
    def _generatePointerUpdates(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                                loadSchedule: List[Dict[str,
                                                        HyperRectangle]], nodeMemoryConstraint: NodeMemoryConstraint,
                                tilingSchedule: TilingSchedule) -> Dict[str, _DMAUpdate]:
        updateDict = {}
        deltaOffsets = {}

        for idx, loadStep in enumerate(loadSchedule):
            for stepIdx, (key, rect) in enumerate(loadStep.items()):

                if key in tilingSchedule.outputBaseOffsets.keys():
                    baseOffsets = tilingSchedule.outputBaseOffsets[key]
                    direction = "FromL1"
                else:
                    baseOffsets = tilingSchedule.inputBaseOffsets[key]
                    direction = "ToL1"

                if key not in updateDict.keys():
                    updateDict[key] = []
                if key not in deltaOffsets.keys():
                    deltaOffsets[key] = 0

                referenceBuffer = ctxt.lookup(ctxt.lookup(operatorRepresentation[key])._referenceName)
                l1Buffer = ctxt.lookup(operatorRepresentation[key])

                finalMemoryLevel = TilingCodeGeneration.isFinalMemoryLevel(nodeMemoryConstraint, l1Buffer)

                if (f"in{stepIdx}_perm" in operatorRepresentation
                        and key in tilingSchedule.inputBaseOffsets.keys()) and (finalMemoryLevel == False):
                    perm = operatorRepresentation[f"in{stepIdx}_perm"]
                    struct, _, _ = AutoTransposeUtils.generateTransposedDMAStruct(ctxt, rect, direction, perm,
                                                                                  l1Buffer.name,
                                                                                  l1Buffer._referenceName)

                    _invPerm = _invertPermutation(perm)
                    _rect = copy.copy(rect)
                    _referenceBuffer = copy.copy(referenceBuffer)
                    _rect.offset = _permuteList(rect.offset, _invPerm)
                    _rect.dims = _permuteList(rect.dims, _invPerm)
                    _referenceBuffer.shape = _permuteList(referenceBuffer.shape, _invPerm)

                    accOffset = calculateRectangleOffset(_rect, _referenceBuffer)

                else:
                    struct = cls._rectToDMAStruct(ctxt, rect, direction, l1Buffer.name, l1Buffer._referenceName,
                                                  finalMemoryLevel)
                    accOffset = calculateRectangleOffset(rect, referenceBuffer)

                length_1d_copy = struct.value['length_1d_copy'].value
                number_of_1d_copies = struct.value['number_of_1d_copies'].value
                number_of_2d_copies = struct.value['number_of_2d_copies'].value
                stride_1d = struct.value['stride_1d'].value
                stride_2d = struct.value['stride_2d'].value
                mchan_cmd = struct.value['mchan_cmd'].value

                lIdx = idx % len(baseOffsets)

                sol = _DMAUpdate(accOffset, baseOffsets[lIdx], length_1d_copy, number_of_1d_copies, number_of_2d_copies,
                                 stride_1d, stride_2d, mchan_cmd)

                deltaOffsets[key] = accOffset
                updateDict[key].append(sol)

        return updateDict

    @classmethod
    def _rectToDMAStruct(cls, ctxt: NetworkContext, rectangle: HyperRectangle, direction: Literal["ToL1", "FromL1"],
                         L1Name: str, L2Name: str, finalMemoryLevel: bool) -> PULPStructDataTypes.DMA_copy:

        referenceBuffer = ctxt.lookup(L2Name)

        rect, referenceRect = minimizeRectangleDims(rectangle, referenceBuffer)
        assert len(rect.dims) <= 3, "PULP: Only 2D transfers are supported!"

        if direction == "ToL1":
            _dir = 1
        else:
            _dir = 0

        length_1d_copy = rect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

        number_of_1d_copies = 1
        stride_1d = 0

        if len(rect.dims) > 1:
            number_of_1d_copies = rect.dims[-2]
            stride_1d = referenceRect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

            if not finalMemoryLevel:
                stride_1d = length_1d_copy

        number_of_2d_copies = 1
        stride_2d = 0

        if len(rect.dims) > 2:
            number_of_2d_copies = rect.dims[-3]
            stride_2d = referenceRect.dims[-2] * stride_1d

        length_2d_copy = number_of_1d_copies * length_1d_copy
        mchan_flags = _dir + 0x2 + 0x8
        if number_of_1d_copies > 1 or number_of_2d_copies > 1:
            mchan_flags += 0x4
        mchan_cmd = length_2d_copy + (mchan_flags << 17)

        struct = PULPStructDataTypes.DMA_copy(
            {
                "ext": referenceBuffer.name,
                "loc": L1Name,
                "hwc_to_chw": 0,
                "stride_2d": stride_2d,
                "number_of_2d_copies": number_of_2d_copies,
                "stride_1d": stride_1d,
                "number_of_1d_copies": number_of_1d_copies,
                "length_1d_copy": length_1d_copy,
                "mchan_cmd": mchan_cmd,
                "dir": _dir,
                "tid": 0
            }, ctxt)

        return struct

    def _hoistConstantAndReference(self,
                                   ctxt: NetworkContext,
                                   constBuf: ConstantBuffer,
                                   operatorRepresentation: OperatorRepresentation,
                                   nodeName: str,
                                   operatorRepresentationName: str,
                                   immediateType: Optional[Type[Immediate]] = None) -> Tuple[NetworkContext, Dict]:

        if immediateType is None:
            _type = PointerClass(BasicDataTypes.int32_t)
        else:
            _type = PointerClass(immediateType)

        name = constBuf.name

        ctxt.add(constBuf, "global")
        constBuf._type = _type
        constBuf._instance = constBuf._type(name, ctxt)
        constBuf._users = [nodeName]
        constBuf._memoryLevel = self.targetMemLevel

        refName = name + "_ref"
        reference = ctxt.hoistReference(name, refName)
        ctxt.lookup(reference)._memoryLevel = self.targetMemLevel

        operatorRepresentation[operatorRepresentationName] = refName

        return ctxt, operatorRepresentation

    def _hoistDMAUpdates(self, ctxt: NetworkContext, tensorName: str, updateList: List[_DMAUpdate],
                         operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        operatorRepresentation = operatorRepresentation.copy()

        nodeName = operatorRepresentation['nodeName']

        offsetList = []
        mchanCmdList = []
        len1dList = []
        num1dList = []
        num2dList = []
        stride1dList = []
        stride2dList = []
        for update in updateList:
            offsetList.append(int(update.extOffset))
            mchanCmdList.append(int(update.mchan_cmd))
            len1dList.append(int(update.length_1d_copy))
            num1dList.append(int(update.number_of_1d_copies))
            num2dList.append(int(update.number_of_2d_copies))
            stride1dList.append(int(update.stride_1d))
            stride2dList.append(int(update.stride_2d))

        dmaName = self._DMAStructName(tensorName, nodeName)
        operatorRepresentation['stateReference'] = dmaName
        operatorRepresentation['tileNum'] = "TILING_I"
        operatorRepresentation['extPtr'] = ctxt.lookup(operatorRepresentation[tensorName])._referenceName

        namePrefix = self.prefix + f"{nodeName}_{tensorName}"

        name = namePrefix + "_offset"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], offsetList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(ctxt, cb, operatorRepresentation, nodeName,
                                                                       'extOffsetPtr')

        name = namePrefix + "_mchan_cmd"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], mchanCmdList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'mchanCmdPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['mchan_cmd'])

        name = namePrefix + "_length_1d_copy"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], len1dList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'length1dPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['length_1d_copy'])

        name = namePrefix + "_number_of_1d_copies"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], num1dList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'number1dPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['number_of_1d_copies'])

        name = namePrefix + "_number_of_2d_copies"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], num2dList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'number2dPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['number_of_2d_copies'])

        name = namePrefix + "_stride_1d"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], stride1dList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'stride1dPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['stride_1d'])

        name = namePrefix + "_stride_2d"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], stride2dList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'stride2dPtr',
            PULPStructDataTypes.DMA_copy.structTypeDict['stride_2d'])

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
            updates.append(CodeSnippet(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateIngressPointerUpdates(
            self, nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, List[CodeSnippet]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = self._generatePointerUpdates(ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
                                                  nodeMemoryConstraint, tilingSchedule)

        for key, updateList in updateDict.items():

            newCtxt, newNodeRep = self._hoistDMAUpdates(newCtxt, key, updateList, operatorRepresentation)
            updates.append(CodeSnippet(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateVariableUpdates(self, tilingSchedule: TilingSchedule, variableReplacement: VariableReplacementScheme,
                                 ctxt: NetworkContext,
                                 operatorRepresentation: OperatorRepresentation) -> List[CodeSnippet]:

        updates = []

        for key in variableReplacement.perTileReplacements.keys():

            buf = ctxt.lookup(operatorRepresentation[key])
            reference = str(buf._instance)

            updates.append(
                CodeSnippet(self._updateReferenceTemplate, {
                    "reference": reference,
                    "tileNum": "TILING_I",
                    "baseReference": buf._referenceName
                }))

        return updates

    def _generateDMACode(self, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
                         operatorRepresentation: OperatorRepresentation, loadSchedule: List[Dict[str, HyperRectangle]],
                         direction: Literal["ToL1", "FromL1"]) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        DMATransferCalls = []
        DMAWaitStatements = []

        allNumTransfers = AutoTransposeUtils.allNumTransfers(ctxt, operatorRepresentation, loadSchedule, direction)

        transferNodeRep = {}

        if allNumTransfers != []:

            dimLens = []

            for dim in range(len(allNumTransfers[0])):
                dimVec = [transfer[dim] for transfer in allNumTransfers]
                namePrefix = operatorRepresentation["nodeName"] + "_"
                vecName = f"dimLen_{dim}"

                cb = ctxt.ConstantBuffer(namePrefix + vecName, [len(dimVec)], dimVec)
                ctxt, transferNodeRep = self._hoistConstantAndReference(ctxt, cb, transferNodeRep,
                                                                        operatorRepresentation['nodeName'], vecName)

                dimLens.append(str(cb._instance))

            transferNodeRep['nodeName'] = operatorRepresentation['nodeName']
            transferNodeRep['dimLens'] = dimLens
            transferNodeRep['tileNum'] = "TILING_I"

        loadStep = loadSchedule[0]

        for idx, (key, rectangle) in enumerate(loadStep.items()):

            permName = f"in{idx}_perm"

            externalPtr = ctxt.lookup(ctxt.lookup(operatorRepresentation[key])._referenceName)
            internalPtr = ctxt.lookup(operatorRepresentation[key])

            tensorName = key
            nodeName = operatorRepresentation['nodeName']
            dmaName = self._DMAStructName(tensorName, nodeName)

            transferNodeRep = {
                **transferNodeRep,
                **{
                    'innerTilePtr': str(internalPtr._instance),
                    "outerTilePtr": str(externalPtr._instance),
                    "stateReference": dmaName
                }
            }

            if permName in operatorRepresentation and direction == "ToL1":
                perm = operatorRepresentation[permName]
                struct, remainderStrides, numTransfers = AutoTransposeUtils.generateTransposedDMAStruct(
                    ctxt, rectangle, direction, perm, internalPtr.name, externalPtr.name)
                locStride = np.prod(
                    rectangle.dims) // np.prod(numTransfers) * (externalPtr._type.referencedType.typeWidth // 8)

                transferNodeRep['tensorName'] = operatorRepresentation[key]

                transferNodeRep = {**transferNodeRep, **{"remainderStrides": remainderStrides, "locStride": locStride}}

            else:
                finalMemoryLevel = TilingCodeGeneration.isFinalMemoryLevel(nodeMemoryConstraint, internalPtr)

                struct = self._rectToDMAStruct(ctxt, rectangle, direction, internalPtr.name, externalPtr.name,
                                               finalMemoryLevel)

            transferNodeRep["stateStruct"] = struct
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.DMA_copy)
            ctxt.lookup(dmaName)._users += [operatorRepresentation['nodeName']]

            if permName in operatorRepresentation and direction == "ToL1":

                DMATransferCalls.append(CodeSnippet(self._iteratedMoveTileInTemplate, transferNodeRep))
            else:
                DMATransferCalls.append(CodeSnippet(self._moveTileInTemplate, transferNodeRep))

            DMAWaitStatements.append(CodeSnippet(self._blockTileInTemplate, transferNodeRep))

        return DMATransferCalls, DMAWaitStatements

    def _generateIngressDMACode(
            self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        importLoadStep = tilingSchedule.inputLoadSchedule
        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateDMACode(nodeMemoryConstraint, ctxt,
                                                                                  operatorRepresentation,
                                                                                  importLoadStep, "ToL1")
        return ingressDMATransferCalls, ingressDMAWaitStatements

    def _generateEgressDMACode(
            self, tilingSchedule: TilingSchedule, nodeMemoryConstraint: NodeMemoryConstraint, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        exportLoadStep = tilingSchedule.outputLoadSchedule
        egressDMATransferCalls, egressDMAWaitStatements = self._generateDMACode(nodeMemoryConstraint, ctxt,
                                                                                operatorRepresentation, exportLoadStep,
                                                                                "FromL1")

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

        setupStatements = [CodeSnippet(self._initDMATemplate, {"channelName": "dma_channel"})]
        setupStatements += [
            CodeSnippet(self._setDMAChannelTemplate, {
                **transaction.operatorRepresentation, "channelName": "dma_channel"
            }) for transaction in ingressDMAUpdates + egressDMAUpdates
        ]

        teardownStatements = [
            CodeSnippet(self._releaseDMATemplate,
                        {"stateReference": ingressDMAUpdates[0].operatorRepresentation["stateReference"]})
        ]

        variableUpdates = self._generateVariableUpdates(tilingSchedule, variableReplacement, ctxt,
                                                        operatorRepresentation)

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'] + "_L2",
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

        # SCHEREMO: hoist numTiles

        offsetLists = list({**flatTilingSchedule.inputBaseOffsets, **flatTilingSchedule.outputBaseOffsets}.values())

        if len(offsetLists) == 0:
            return ctxt, executionBlock, False

        for offsetList in offsetLists:
            if not len(offsetList) == 1:
                return ctxt, executionBlock, False

        operatorRepresentation["numTiles"] = self._hoistNumTiles(ctxt, operatorRepresentation['nodeName'],
                                                                 tilingSchedules)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)


class PULPClusterTilingGenerationSB(PULPClusterTilingSB, SingleBufferingTilingMixIn):
    pass


class ProfilingPULPClusterTilingGenerationSB(PULPClusterTilingSB, ProfilingSingleBufferingTilingMixIn):
    pass
