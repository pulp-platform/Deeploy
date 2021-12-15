# ----------------------------------------------------------------------
#
# File: TilingCodegen.py
#
# Last edited: 11.10.2023
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

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Type

import numpy as np

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint, NodeMemoryConstraint


@dataclass
class MemoryTransfer():
    source: MemoryConstraint
    destination: MemoryConstraint


@dataclass
class HyperRectangle():
    # position of the hyperrectangle in feature map space
    offset: Tuple[int, ...]
    # size of the hyperrectangle
    dims: Tuple[int, ...]

    def __init__(self, offset: Tuple[int, ...], dims: Tuple[int, ...]):
        assert len(offset) == len(
            dims), f"HyperRectangle offset and dims for mismatching dimensions {offset} and {dims}"

        self.offset = offset
        self.dims = dims


@dataclass
class AbsoluteHyperRectangle:
    rectangle: HyperRectangle
    absoluteOffset: Tuple[int, ...]

    def __init__(self, rectangle: HyperRectangle, absoluteOffset: Tuple[int, ...]):
        assert len(absoluteOffset) == len(
            rectangle.offset
        ), f"AsoluteHyperRectangle's absoluteOffset and rectangle's offset for mismatching dimensions {absoluteOffset} and {rectangle.offset}"

        self.rectangle = rectangle
        self.absoluteOffset = absoluteOffset


@dataclass
class TilingSchedule():
    # the places to store input tiles
    # Should have length numTiles
    inputBaseOffsets: Dict[str, List[int]]

    # the places to store output tiles
    # Should have length numTiles
    outputBaseOffsets: Dict[str, List[int]]

    # the hypercubes to load in each step
    # Should have length numTiles
    inputLoadSchedule: List[Dict[str, HyperRectangle]]

    # the hypercubes to store in each step
    # Should have length numTiles
    outputLoadSchedule: List[Dict[str, HyperRectangle]]

    def __init__(self, inputBaseOffsets: Dict[str, List[int]], outputBaseOffsets: Dict[str, List[int]],
                 inputLoadSchedule: List[Dict[str, HyperRectangle]], outputLoadSchedule: List[Dict[str,
                                                                                                   HyperRectangle]]):

        # assert len(inputLoadSchedule) == len(outputLoadSchedule), "Didn't get equal amount of input and output tiles!"

        for scheduleStep in inputLoadSchedule:
            for key in inputBaseOffsets:
                assert key in scheduleStep.keys(), f"Key {key} is not in scheduleStep {scheduleStep}"

        for scheduleStep in outputLoadSchedule:
            for key in outputBaseOffsets:
                assert key in scheduleStep.keys(), f"Key {key} is not in scheduleStep {scheduleStep}"

        self.inputBaseOffsets = inputBaseOffsets
        self.outputBaseOffsets = outputBaseOffsets
        self.inputLoadSchedule = inputLoadSchedule
        self.outputLoadSchedule = outputLoadSchedule

    def __repr__(self) -> str:
        outStr = ""
        outStr += f"inputBaseOffsets: \n{str(self.inputBaseOffsets)} \n"
        outStr += f"outputBaseOffsets: \n{str(self.outputBaseOffsets)} \n"

        inSched = ("\n").join([str(step) for step in self.inputLoadSchedule])
        outSched = ("\n").join([str(step) for step in self.outputLoadSchedule])

        outStr += f"inputLoadSchedule: \n{inSched} \n"
        outStr += f"outputLoadSchedule: \n{outSched} \n"

        return outStr

    def __add__(self, other: TilingSchedule) -> TilingSchedule:

        assert isinstance(other, TilingSchedule), f"Other {other} is not a TilingSchedule"

        for key in self.inputBaseOffsets.keys():
            assert key in other.inputBaseOffsets.keys(), f"Other {other} has no key {key}"
        for key in self.outputBaseOffsets.keys():
            assert key in other.outputBaseOffsets.keys(), f"Other {other} has no key {key}"

        for key in other.inputBaseOffsets.keys():
            assert key in self.inputBaseOffsets.keys(), f"Other {other} has no key {key}"
        for key in other.outputBaseOffsets.keys():
            assert key in self.outputBaseOffsets.keys(), f"Other {other} has no key {key}"

        new = TilingSchedule(self.inputBaseOffsets.copy(), self.outputBaseOffsets.copy(), self.inputLoadSchedule.copy(),
                             self.outputLoadSchedule.copy())

        new.inputLoadSchedule += other.inputLoadSchedule
        new.outputLoadSchedule += other.outputLoadSchedule

        return new


@dataclass
class VariableReplacementScheme():
    perTileReplacements: Dict[str, List]
    replacementTypes: Dict[str, Type[Pointer]]

    def __init__(self, perTileReplacements: Dict[str, List], replacementTypes: Dict[str, Type[Pointer]]):
        assert len(perTileReplacements.keys()) == len(
            replacementTypes.keys()), "Exactly all replacements must have one type"

        for key in perTileReplacements.keys():
            assert key in replacementTypes.keys(), "Keys must match!"

        self.perTileReplacements = perTileReplacements
        self.replacementTypes = replacementTypes

    def __add__(self, other: VariableReplacementScheme) -> VariableReplacementScheme:

        assert isinstance(other, VariableReplacementScheme), f"Other {other} is not a VariableReplacementScheme"

        for key in self.perTileReplacements.keys():
            assert key in other.perTileReplacements.keys(), f"key {key} not in other {other}!"
        for key in self.replacementTypes.keys():
            assert key in other.replacementTypes.keys(), f"key {key} not in other {other}!"

        for key in other.perTileReplacements.keys():
            assert key in self.perTileReplacements.keys(), f"key {key} not in other {other}!"
        for key in other.replacementTypes.keys():
            assert key in self.replacementTypes.keys(), f"key {key} not in other {other}!"

        new = VariableReplacementScheme(self.perTileReplacements.copy(), self.replacementTypes.copy())
        for key in self.perTileReplacements.keys():
            new.perTileReplacements[key] += other.perTileReplacements[key]

        return new


def minimizeVariableReplacement(
        scheme: VariableReplacementScheme,
        operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, Dict]:
    newPerTileRep = {}
    newRepTypes = {}

    for key, value in scheme.perTileReplacements.items():
        if len(set(value)) > 1:
            newPerTileRep[key] = scheme.perTileReplacements[key]
            newRepTypes[key] = scheme.replacementTypes[key]
        else:
            operatorRepresentation[key] = value[0]

    return VariableReplacementScheme(newPerTileRep, newRepTypes), operatorRepresentation


def minimizeRectangleDims(hyperRectangle: HyperRectangle,
                          referenceBuffer: VariableBuffer) -> Tuple[HyperRectangle, HyperRectangle]:

    rectDims = hyperRectangle.dims
    rectOffset = hyperRectangle.offset
    shape = referenceBuffer.shape
    newDims: List[int] = []
    newOffset: List[int] = []

    newBaseline = []

    reversedRectOffset = list(reversed(rectOffset))

    # SCHEREMO: Collapse dimensions right to left
    acc = 0
    for idx, (tileDim, bufDim) in enumerate(zip(reversed(rectDims), reversed(shape))):

        if tileDim == bufDim:
            assert reversedRectOffset[idx] == 0, "Can't not tile a dimension and have an offset, tf"

        # SCHEREMO: Collapse if equal
        if tileDim == bufDim and acc != 0:
            acc *= tileDim
        elif tileDim == bufDim and acc == 0:
            acc = tileDim
        elif tileDim != bufDim and acc != 0:
            newDims.insert(0, acc * tileDim)
            newBaseline.insert(0, acc * bufDim)
            newOffset.insert(0, acc * reversedRectOffset[idx])
            acc = 0
        else:
            newDims.insert(0, tileDim)
            newBaseline.insert(0, bufDim)
            newOffset.insert(0, reversedRectOffset[idx])

    if acc > 1:
        newDims.insert(0, acc)
        newBaseline.insert(0, acc)
        newOffset.insert(0, acc * reversedRectOffset[idx])

    # JUNGVI: If the function collapsed all dimensions of the tensor, set it to dim 1 and offset 0
    if len(newDims) == 0:
        newDims = [1]
        newBaseline = [1]
        newOffset = [0]

    newRect = HyperRectangle(tuple(newOffset), tuple(newDims))
    newBaseline = HyperRectangle(tuple([0] * len(newOffset)), tuple(newBaseline))

    return newRect, newBaseline


def calculateRectangleOffset(hyperRectangle: HyperRectangle, referenceBuffer: VariableBuffer) -> int:

    minimalRect, baselineRect = minimizeRectangleDims(hyperRectangle, referenceBuffer)

    offsetMult = [1]
    for dim in reversed(baselineRect.dims[1:]):
        offsetMult.insert(0, dim * np.prod(offsetMult))

    accOffset = 0
    for offsetIdx, mult in zip(minimalRect.offset, offsetMult):
        accOffset += offsetIdx * mult

    return int(accOffset * (referenceBuffer._type.referencedType.typeWidth // 8))


def extractTilingTransfer(tilingSolution: NodeMemoryConstraint, targetMemLevel: str,
                          tensorName: str) -> Optional[MemoryTransfer]:

    for name, constraint in tilingSolution.tensorMemoryConstraints.items():
        if not name == tensorName:
            continue

        sourceIdx = 0

        for idx, memConstraint in enumerate(constraint.memoryConstraints.values()):
            if memConstraint.memoryLevel != targetMemLevel:
                continue

            sourceIdx = idx
            targetIdx = idx - 1

            if sourceIdx == 0:
                return None

            return MemoryTransfer(
                list(constraint.memoryConstraints.values())[targetIdx],
                list(constraint.memoryConstraints.values())[sourceIdx])

    raise RuntimeError(f"{tensorName} not found in tilingSolution!")


def computeHyperRectangleList(memTrans: MemoryTransfer) -> List[HyperRectangle]:

    def nextElement(idxVec: List[int], targetVector: List[int]) -> Optional[List[int]]:
        nextIdx = []

        countUp = True
        for vecIdx, maxIdx in zip(reversed(idxVec), reversed(targetVector)):
            if countUp:
                if vecIdx == maxIdx:
                    nextIdx.append(1)
                else:
                    nextIdx.append(vecIdx + 1)
                    countUp = False
            else:
                nextIdx.append(vecIdx)

        nextIdx.reverse()

        if countUp:
            return None

        return nextIdx

    def calculateCost(idxVec: Iterable[int], smallShape: Tuple[int]) -> List[int]:
        outVec = []
        for idx, step in zip(idxVec, smallShape):
            outVec.append((idx - 1) * step)

        return outVec

    def calculateDim(idxVec: List[int], numTiles: List[int], smallShape: Tuple[int],
                     largeShape: Tuple[int]) -> List[int]:

        dimVec = []

        for idx, (vecIdx, maxIdx) in enumerate(zip(idxVec, numTiles)):
            if vecIdx != maxIdx:
                dimVec.append(smallShape[idx])
                continue
            if largeShape[idx] % smallShape[idx] == 0:
                dimVec.append(smallShape[idx])
                continue
            dimVec.append(largeShape[idx] % smallShape[idx])

        return dimVec

    src = memTrans.source
    dst = memTrans.destination

    largeShape = src.shape
    smallShape = dst.shape

    assert largeShape is not None, "Transfer shapes cannot be undefined!"
    assert smallShape is not None, "Transfer shapes cannot be undefined!"

    assert len(smallShape) == len(
        largeShape), f"Source and target of memory transfer {memTrans} don't have the same number of dimensions!"
    for idx, (dim1, dim2) in enumerate(zip(smallShape, largeShape)):
        assert dim1 <= dim2, f"Large shape is smaller in dimension {idx}"

    totNumTiles = 1
    numTiles: List[int] = []

    for (dim1, dim2) in zip(smallShape, largeShape):
        totNumTiles *= np.ceil(dim2 / dim1)
        numTiles.append(int(np.ceil(dim2 / dim1)))

    cubeList: List[HyperRectangle] = []
    idxVec = [1] * len(smallShape)

    for i in range(int(totNumTiles)):
        offsetVec = calculateCost(idxVec, smallShape)
        dimVec = calculateDim(idxVec, numTiles, smallShape, largeShape)
        cubeList.append(HyperRectangle(tuple(offsetVec), tuple(dimVec)))

        nextVec = nextElement(idxVec, numTiles)
        if nextVec is None:
            break
        idxVec = nextVec

    return cubeList
