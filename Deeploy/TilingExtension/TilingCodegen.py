# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Sequence, Tuple, Type

import numpy as np

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DeeployTypes import OperatorRepresentation, VariableBuffer
from Deeploy.TilingExtension.MemoryConstraints import MemoryConstraint


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


def minimizeRectangle(rect: HyperRectangle, referenceShape: Sequence[int]) -> Tuple[HyperRectangle, Tuple[int, ...]]:
    minRectShape: List[int] = []
    minRectOffset: List[int] = []
    minReferenceShape: List[int] = []

    # SCHEREMO: Collapse dimensions right to left
    currentCollapsedDim = 1
    for rectDim, rectOffset, referenceDim in zip(reversed(rect.dims), reversed(rect.offset), reversed(referenceShape)):
        if rectDim == referenceDim:
            assert rectOffset == 0, f"Rectangle offset should be zero when the dimensions are the same. Received rectangle {rect} and reference shape {referenceShape}"
            currentCollapsedDim *= rectDim
        else:
            minRectShape.insert(0, currentCollapsedDim * rectDim)
            minReferenceShape.insert(0, currentCollapsedDim * referenceDim)
            minRectOffset.insert(0, currentCollapsedDim * rectOffset)
            currentCollapsedDim = 1

    if currentCollapsedDim > 1 or len(minRectShape) == 0:
        minRectShape.insert(0, currentCollapsedDim)
        minReferenceShape.insert(0, currentCollapsedDim)
        minRectOffset.insert(0, currentCollapsedDim * rect.offset[0])

    return HyperRectangle(tuple(minRectOffset), tuple(minRectShape)), tuple(minReferenceShape)


def padShape(shape: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
    assert rank >= len(
        shape), f"Cannot pad to rank smaller then shape's. Received rank: {rank}, shape rank: {len(shape)}"
    ret = tuple([1] * (rank - len(shape))) + shape
    assert len(ret) == rank
    return ret


def padOffset(offset: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
    assert rank >= len(
        offset), f"Cannot pad to rank smaller then offset's. Received rank: {rank}, offset rank: {len(offset)}"
    ret = tuple([0] * (rank - len(offset))) + offset
    assert len(ret) == rank
    return ret


def padStride(stride: Tuple[int, ...], rank: int, paddingStride: int) -> Tuple[int, ...]:
    assert rank >= len(
        stride), f"Cannot pad to rank smaller then stride's. Received rank: {rank}, stride rank: {len(stride)}"
    ret = tuple([paddingStride] * (rank - len(stride))) + stride
    assert len(ret) == rank
    return ret


def stridesFromShape(shape: Sequence[int]) -> Tuple[int, ...]:
    strides = [1] * len(shape)
    for idx, dim in enumerate(reversed(shape[1:])):
        strides[idx + 1] = strides[idx] * dim
    return tuple(reversed(strides))


def calculateFlatOffset(offsets: Sequence[int], strides: Sequence[int]) -> int:
    assert len(offsets) == len(strides), \
        f"Offsets and strides have to have the same number of dimensions. Length offsets: {len(offsets)}, strides: {len(strides)}"
    return sum(offset * stride for offset, stride in zip(offsets, strides))


def calculateFlatOffsetInBytes(tile: HyperRectangle, referenceBuffer: VariableBuffer) -> int:
    return int(
        calculateFlatOffset(tile.offset, stridesFromShape(referenceBuffer.shape)) *
        (referenceBuffer._type.referencedType.typeWidth // 8))


def computeTileHyperRectangles(memoryTransfer: MemoryTransfer) -> List[HyperRectangle]:
    assert memoryTransfer.source.shape is not None, "Source transfer shape cannot be undefined!"
    assert memoryTransfer.destination.shape is not None, "Destination transfer shape cannot be undefined!"

    assert len(memoryTransfer.source.shape) == len(memoryTransfer.destination.shape), \
    f"Source and target of memory transfer {memoryTransfer} don't have the same number of dimensions!"

    largeShape = memoryTransfer.source.shape
    smallShape = memoryTransfer.destination.shape

    for dimIdx, (dimSizeSmall, dimSizeLarge) in enumerate(zip(smallShape, largeShape)):
        assert dimSizeSmall <= dimSizeLarge, f"smallShape[{dimIdx}] should not be bigger then largeShape[{dimIdx}]. ({dimSizeSmall} > {dimSizeLarge})"

    def nextTileIndex(tileIndexEnd: List[int]) -> Generator[List[int]]:
        tileCount = np.prod(tileIndexEnd)
        tileIndex = [0] * len(tileIndexEnd)
        for _ in range(tileCount):
            yield tileIndex
            for dimIdx, (idx, end) in enumerate(zip(tileIndex, tileIndexEnd)):
                if idx + 1 < end:
                    tileIndex[dimIdx] = idx + 1
                    break
                else:
                    tileIndex[dimIdx] = 0

    tileHyperRectangles = []

    tileIndexEnd = [
        int(np.ceil(dimSizeLarge / dimSizeSmall)) for dimSizeLarge, dimSizeSmall in zip(largeShape, smallShape)
    ]
    for tileIndex in nextTileIndex(tileIndexEnd):
        tileOffset = tuple(dimIdx * dimSizeSmall for dimIdx, dimSizeSmall in zip(tileIndex, smallShape))
        for dimIdx, (dimOffset, dimSizeLarge) in enumerate(zip(tileOffset, largeShape)):
            assert dimOffset >= 0, f"tileOffset[{dimIdx}] shoud not be smaller then zero ({dimOffset} < 0)"
            assert dimOffset < dimSizeLarge, f"tileOffset[{dimIdx}] should not be bigger or equal then largeShape[{dimIdx}] ({dimOffset} >= {dimSizeLarge})"

        tileSize = tuple(
            min(dimSizeSmall, dimSizeLarge - dimOffset)
            for dimSizeSmall, dimSizeLarge, dimOffset in zip(smallShape, largeShape, tileOffset))
        for dimIdx, (dimSize, dimSizeSmall) in enumerate(zip(tileSize, smallShape)):
            assert dimSize > 0, f"tileOffset[{dimIdx}] shoud not be smaller or equal then zero ({dimSize} <= 0)"
            assert dimSize <= dimSizeSmall, f"tileSize[{dimIdx}] should not be bigger then smallShape[{dimIdx}] ({dimSize} > {dimSizeSmall})"

        tileHyperRectangles.append(HyperRectangle(tileOffset, tileSize))

    return tileHyperRectangles
