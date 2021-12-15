# ----------------------------------------------------------------------
#
# File: AutoTransposeUtils.py
#
# Last edited: 11.12.2023
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
from typing import Dict, List, Literal, Tuple

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    _invertPermutation, _permuteList
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.PULPOpen.DataTypes import PULPStructDataTypes
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, minimizeRectangleDims


def _transposedDMAStrides(ctxt: NetworkContext, rectangle: HyperRectangle, direction: Literal["ToL1", "FromL1"],
                          perm: List[int], L1Name: str, L2Name: str) -> Tuple[HyperRectangle, List[int], List[int]]:
    _invPerm = _invertPermutation(perm)
    rectangle = HyperRectangle(_permuteList(rectangle.offset, _invPerm), _permuteList(rectangle.dims, _invPerm))

    contiguousDims = [permIdx == rangeIdx for permIdx, rangeIdx in zip(perm, range(len(perm)))]
    workList = []

    for idx, dim in enumerate(contiguousDims):
        if dim:
            workList.append(rectangle.dims[idx])
        else:
            workList.append(1)

    maxTransferRect = copy.copy(rectangle)
    maxTransferRect.dims = tuple(workList)

    referenceBuffer = copy.copy(ctxt.lookup(L2Name))
    referenceBuffer.shape = _permuteList(referenceBuffer.shape, _invPerm)
    minRect, referenceRect = minimizeRectangleDims(maxTransferRect, referenceBuffer)

    droppedIdx = [
        idx for idx in range(len(perm))
        if (referenceBuffer.shape[idx] == 1 or referenceBuffer.shape[idx] == maxTransferRect.dims[idx])
    ]

    _newPerm = []
    for p in perm:
        if p not in droppedIdx:
            _newPerm.append(p)

    newPerm = []
    for p in _newPerm:
        sub = sum([p > idx for idx in droppedIdx])
        newPerm.append(p - sub)

    strides = [1]
    for dim in reversed(referenceRect.dims[1:]):
        strides.insert(0, strides[0] * dim)

    permStrides = [strides[idx] for idx in newPerm]
    fixedPermStrides = []
    maxStride = 0
    remainderStrides = []
    for stride in reversed(permStrides):
        if stride < maxStride:
            remainderStrides.append(stride)
            continue
        maxStride = max(stride, maxStride)
        fixedPermStrides.insert(0, stride)

    return minRect, fixedPermStrides, remainderStrides


def allNumTransfers(ctxt: NetworkContext, operatorRepresentation: OperatorRepresentation,
                    loadSchedule: List[Dict[str, HyperRectangle]], direction: Literal["ToL1",
                                                                                      "FromL1"]) -> List[List[int]]:

    allNumTransfer: List[List[int]] = []

    for stepIdx, loadStep in enumerate(loadSchedule):
        for idx, (key, rectangle) in enumerate(loadStep.items()):
            permName = f"in{idx}_perm"
            externalPtr = ctxt.lookup(ctxt.lookup(operatorRepresentation[key])._referenceName)
            internalPtr = ctxt.lookup(operatorRepresentation[key])

            tensorName = key
            nodeName = operatorRepresentation['nodeName']

            if permName in operatorRepresentation and direction == "ToL1":
                perm = operatorRepresentation[permName]
                _, _, numTransfers = generateTransposedDMAStruct(ctxt, rectangle, direction, perm, internalPtr.name,
                                                                 externalPtr.name)

                allNumTransfer.append(numTransfers)

    return allNumTransfer


def generateTransposedDMAStruct(ctxt: NetworkContext, rectangle: HyperRectangle, direction: Literal["ToL1", "FromL1"],
                                perm: List[int], L1Name: str,
                                L2Name: str) -> Tuple[PULPStructDataTypes.DMA_copy, List[int], List[int]]:

    #rect, referenceRect = minimizeRectangleDims(maxTransferRect, referenceBuffer)
    referenceBuffer = ctxt.lookup(L2Name)

    _invPerm = _invertPermutation(perm)

    contiguousDims = [permIdx == rangeIdx for permIdx, rangeIdx in zip(perm, range(len(perm)))]
    workList = []

    for idx, dim in enumerate(contiguousDims):
        if dim:
            workList.append(rectangle.dims[idx])
        else:
            workList.append(1)

    maxTransferRect = copy.copy(rectangle)
    maxTransferRect.dims = tuple(workList)

    droppedIdx = [
        idx for idx in range(len(perm))
        if (referenceBuffer.shape[idx] == 1 or referenceBuffer.shape[idx] == maxTransferRect.dims[idx])
    ]

    permOffset = [rectangle.offset[idx] for idx, dims in enumerate(rectangle.dims) if (idx not in droppedIdx)]
    permDims = [dims for idx, dims in enumerate(rectangle.dims) if (idx not in droppedIdx)]

    rect = HyperRectangle(offset = permOffset, dims = permDims)
    minRect, fixedPermStrides, remainderStrides = _transposedDMAStrides(ctxt, rectangle, direction, perm, L1Name,
                                                                        L2Name)

    assert len(fixedPermStrides) <= 2, "PULP: Only 2D transfers are supported!"

    if direction == "ToL1":
        _dir = 1
    else:
        _dir = 0

    length_1d_copy = minRect.dims[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

    if len(fixedPermStrides) >= 1:
        number_of_1d_copies = rect.dims[-1]
        stride_1d = fixedPermStrides[-1] * (referenceBuffer._type.referencedType.typeWidth // 8)

    else:
        number_of_1d_copies = 1
        stride_1d = 0

    if len(fixedPermStrides) >= 2:
        number_of_2d_copies = rect.dims[-2]
        stride_2d = fixedPermStrides[-2] * (referenceBuffer._type.referencedType.typeWidth // 8)
    else:
        number_of_2d_copies = 1
        stride_2d = 0

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
            "dir": _dir,
            "tid": 0
        }, ctxt)

    return struct, remainderStrides, rect.dims[:-len(fixedPermStrides)]
