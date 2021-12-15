# ----------------------------------------------------------------------
#
# File: PULPL3TilingSB.py
#
# Last edited: 19.04.2024
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

import copy
from collections import namedtuple
from typing import Dict, List, Literal, Optional, Tuple, Type

import Deeploy.CommonExtensions.DataTypes as BasicDataTypes
from Deeploy.AbstractDataTypes import Immediate, PointerClass
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
// for (int TILING_I=0; TILING_I<${numTiles}; TILING_I++){
for (int TILING_I=${numTiles}[*${tileIdxPtr}]; TILING_I<${numTiles}[(*${tileIdxPtr})+1]; TILING_I++){
""")

_closeTileLoopTemplate = NodeTemplate("""

// CLOSE TILING LOOP
}
*${tileIdxPtr} += 1;

""")

_moveTileInTemplate = NodeTemplate("""

// IMPORT TILE ${innerTilePtr} from ${outerTilePtr}
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});

""")

_blockTileInTemplate = NodeTemplate("""

// BLOCKING IMPORT TILE ${innerTilePtr}
pi_cl_ram_copy_wait(&${stateReference});
""")

_moveTileOutTemplate = NodeTemplate("""

// EXPORT TILE ${innerTilePtr} to ${outerTilePtr}
pi_cl_ram_copy_2d(get_ram_ptr(), ${stateReference}.pi_ram_addr, ${stateReference}.addr, ${stateReference}.size, ${stateReference}.stride, ${stateReference}.length, ${stateReference}.ext2loc, &${stateReference});

""")

_blockTileOutTemplate = NodeTemplate("""

// BLOCKING EXPORT TILE ${innerTilePtr}
pi_cl_ram_copy_wait(&${stateReference});

""")

_updateDMATransferStructTemplate = NodeTemplate("""

// UPDATE DMA STRUCT ${stateReference}
${stateReference}.pi_ram_addr = ((char*)${extPtr}) + ${extOffsetPtr}[${tileNum}];
${stateReference}.size = ${length1dPtr}[${tileNum}];
${stateReference}.length = ${number1dPtr}[${tileNum}];

""")

# ${stateReference}.number_of_2d_copies = ${number2dPtr}[${tileNum}];

_updateReferenceTemplate = NodeTemplate("""

// UPDATE VARIABLE ${reference}
*${reference} = ${baseReference}[${tileNum}];
""")

# ADD NUM TRANSFERS VARIABLE

_DMAUpdate = namedtuple("_DMAUpdate", "extOffset locOffset length_1d_copy number_of_1d_copies number_of_2d_copies")


class PULPL3TilingSB(TilingCodeGeneration):

    _prefix = "TILING_REPLACED_"

    _openTileLoopTemplate = _openTileLoopTemplate
    _closeTileLoopTemplate = _closeTileLoopTemplate

    _moveTileInTemplate = _moveTileInTemplate
    _blockTileInTemplate = _blockTileInTemplate

    _moveTileOutTemplate = _moveTileOutTemplate
    _blockTileOutTemplate = _blockTileOutTemplate

    _updateDMATransferStructTemplate = _updateDMATransferStructTemplate
    _updateReferenceTemplate = _updateReferenceTemplate

    @property
    def prefix(self):
        return self._prefix + self.targetMemLevel + "_"

    def _DMAStructName(self, tensorName: str, nodeName: str) -> str:
        return f"{self.prefix}_DMA_{nodeName}_{tensorName}"

    @classmethod
    def _generatePointerUpdates(cls, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                                loadSchedule: List[Dict[str, HyperRectangle]],
                                tilingSchedule: TilingSchedule) -> Dict[str, _DMAUpdate]:
        updateDict = {}
        deltaOffsets = {}

        for idx, loadStep in enumerate(loadSchedule):
            for stepIdx, (key, rect) in enumerate(loadStep.items()):

                if key in tilingSchedule.outputBaseOffsets.keys():
                    baseOffsets = tilingSchedule.outputBaseOffsets[key]
                    direction = "FromL2"
                else:
                    baseOffsets = tilingSchedule.inputBaseOffsets[key]
                    direction = "ToL2"

                if key not in updateDict.keys():
                    updateDict[key] = []
                if key not in deltaOffsets.keys():
                    deltaOffsets[key] = 0

                referenceBuffer = ctxt.lookup(ctxt.lookup(operatorRepresentation[key])._referenceName)
                l1Buffer = ctxt.lookup(operatorRepresentation[key])

                struct = cls._rectToDMAStruct(ctxt, rect, direction, l1Buffer.name, l1Buffer._referenceName)
                accOffset = calculateRectangleOffset(rect, referenceBuffer)

                length_1d_copy = struct.value['size'].value
                number_of_1d_copies = struct.value['length'].value

                lIdx = idx % len(baseOffsets)

                sol = _DMAUpdate(accOffset, baseOffsets[lIdx], length_1d_copy, number_of_1d_copies, 0)

                deltaOffsets[key] = accOffset
                updateDict[key].append(sol)

        return updateDict

    @classmethod
    def _rectToDMAStruct(cls, ctxt: NetworkContext, rectangle: HyperRectangle, direction: Literal["ToL2", "FromL2"],
                         L1Name: str, L2Name: str) -> PULPStructDataTypes.pi_cl_ram_req_t:

        referenceBuffer = ctxt.lookup(L2Name)

        rect, referenceRect = minimizeRectangleDims(rectangle, referenceBuffer)
        assert len(rect.dims) <= 2, "PULP: Only 2D transfers are supported!"

        if direction == "ToL2":
            _dir = 1
        else:
            _dir = 0

        length_1d_copy = rect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

        if len(rect.dims) > 1:
            number_of_1d_copies = rect.dims[-2]
            stride_1d = referenceRect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)
        else:
            number_of_1d_copies = 1
            stride_1d = 0

        struct = PULPStructDataTypes.pi_cl_ram_req_t(
            {
                "pi_ram_addr": referenceBuffer.name,
                "addr": L1Name,
                "stride": stride_1d,
                "length": length_1d_copy,
                "size": number_of_1d_copies * length_1d_copy,
                "ext2loc": _dir,
                "is_2d": 1
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

        constBuf._users = [nodeName]
        constBuf._memoryLevel = self.targetMemLevel

        refName = ctxt.hoistConstantAndReference(constBuf, _type)

        operatorRepresentation[operatorRepresentationName] = refName

        return ctxt, operatorRepresentation

    def _hoistDMAUpdates(self, ctxt: NetworkContext, tensorName: str, updateList: List[_DMAUpdate],
                         operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict]:

        operatorRepresentation = operatorRepresentation.copy()

        nodeName = operatorRepresentation['nodeName']

        offsetList = []
        len1dList = []
        num1dList = []
        num2dList = []
        for update in updateList:
            offsetList.append(int(update.extOffset))
            len1dList.append(int(update.length_1d_copy))
            num1dList.append(int(update.number_of_1d_copies))
            num2dList.append(int(update.number_of_2d_copies))

        dmaName = self._DMAStructName(tensorName, nodeName)
        operatorRepresentation['stateReference'] = dmaName
        operatorRepresentation['tileNum'] = "TILING_I"
        operatorRepresentation['extPtr'] = ctxt.lookup(operatorRepresentation[tensorName])._referenceName

        namePrefix = self.prefix + f"{nodeName}_{tensorName}"

        name = namePrefix + "_offset"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], offsetList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(ctxt, cb, operatorRepresentation, nodeName,
                                                                       'extOffsetPtr')

        name = namePrefix + "_length_1d_copy"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], len1dList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'length1dPtr',
            PULPStructDataTypes.pi_cl_ram_req_t.structTypeDict['size'])

        name = namePrefix + "_number_of_1d_copies"
        cb = ctxt.ConstantBuffer(name, [len(updateList)], num1dList)
        ctxt, operatorRepresentation = self._hoistConstantAndReference(
            ctxt, cb, operatorRepresentation, nodeName, 'number1dPtr',
            PULPStructDataTypes.pi_cl_ram_req_t.structTypeDict['length'])

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
            updates.append(CodeSnippet(self._updateDMATransferStructTemplate, newNodeRep))

        return newCtxt, updates

    def _generateIngressPointerUpdates(
            self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, List[CodeSnippet]]:

        updates = []
        newCtxt = ctxt.copy()

        updateDict = self._generatePointerUpdates(ctxt, operatorRepresentation, tilingSchedule.inputLoadSchedule,
                                                  tilingSchedule)

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

    def _generateDMACode(self, ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                         loadSchedule: List[Dict[str, HyperRectangle]],
                         direction: Literal["ToL2", "FromL2"]) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        DMATransferCalls = []
        DMAWaitStatements = []

        allNumTransfers = AutoTransposeUtils.allNumTransfers(ctxt, operatorRepresentation, loadSchedule, direction)

        transferNodeRep = {}

        loadStep = loadSchedule[0]

        for idx, (key, rectangle) in enumerate(loadStep.items()):

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

            struct = self._rectToDMAStruct(ctxt, rectangle, direction, internalPtr.name, externalPtr.name)
            transferNodeRep["stateStruct"] = struct
            _ = ctxt.hoistStruct(struct, dmaName, PULPStructDataTypes.pi_cl_ram_req_t)
            ctxt.lookup(dmaName)._users += [operatorRepresentation['nodeName']]

            DMATransferCalls.append(CodeSnippet(self._moveTileInTemplate, transferNodeRep))

            DMAWaitStatements.append(CodeSnippet(self._blockTileInTemplate, transferNodeRep))

        return DMATransferCalls, DMAWaitStatements

    def _generateIngressDMACode(
            self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        importLoadStep = tilingSchedule.inputLoadSchedule
        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateDMACode(ctxt, operatorRepresentation,
                                                                                  importLoadStep, "ToL2")
        return ingressDMATransferCalls, ingressDMAWaitStatements

    def _generateEgressDMACode(
            self, tilingSchedule: TilingSchedule, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[List[CodeSnippet], List[CodeSnippet]]:

        exportLoadStep = tilingSchedule.outputLoadSchedule
        egressDMATransferCalls, egressDMAWaitStatements = self._generateDMACode(ctxt, operatorRepresentation,
                                                                                exportLoadStep, "FromL2")

        return egressDMATransferCalls, egressDMAWaitStatements

    def _tilingLoop(self, ctxt: NetworkContext, executionBlock: ExecutionBlock,
                    nodeMemoryConstraint: NodeMemoryConstraint, tilingSchedule: TilingSchedule,
                    variableReplacement: VariableReplacementScheme,
                    operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, ExecutionBlock, bool]:

        tileIdxPtr = self._hoistTileIdxPtr(ctxt, operatorRepresentation)

        ingressDMATransferCalls, ingressDMAWaitStatements = self._generateIngressDMACode(
            tilingSchedule, ctxt, operatorRepresentation)

        egressDMATransferCalls, egressDMAWaitStatements = self._generateEgressDMACode(
            tilingSchedule, ctxt, operatorRepresentation)

        ctxt, ingressDMAUpdates = self._generateIngressPointerUpdates(tilingSchedule, ctxt, operatorRepresentation)
        ctxt, egressDMAUpdates = self._generateEgressPointerUpdates(tilingSchedule, ctxt, operatorRepresentation)

        setupStatements: List[CodeSnippet] = []
        teardownStatements: List[CodeSnippet] = []
        variableUpdates: List[CodeSnippet] = []

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

        metaInfo = TilingMetaInfo(nodeName = operatorRepresentation['nodeName'],
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
            if not len(offsetList) == 1:
                return ctxt, executionBlock, False

        operatorRepresentation["numTiles"] = self._hoistNumTiles(ctxt, operatorRepresentation['nodeName'],
                                                                 tilingSchedules)

        return self._tilingLoop(ctxt, executionBlock, nodeMemoryConstraint, flatTilingSchedule, variableReplacement,
                                operatorRepresentation)


class PULPL3TilingGenerationSB(PULPL3TilingSB, SingleBufferingTilingMixIn):
    pass


class ProfilingPULPL3TilingGenerationSB(PULPL3TilingSB, ProfilingSingleBufferingTilingMixIn):
    pass
