# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
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
    """
    Represents a memory transfer operation between two memory levels.

    This dataclass encapsulates the source and destination memory constraints
    for a memory transfer operation in the tiling system, defining where data
    is transferred from and to in the memory hierarchy.

    Attributes
    ----------
    source : MemoryConstraint
        The source memory constraint defining the memory level data is
        transferred from.
    destination : MemoryConstraint
        The destination memory constraint defining the memory level data is
        transferred to.

    Notes
    -----
    This class is used in conjunction with memory hierarchies to define
    data movement patterns during tiled neural network execution.
    """
    source: MemoryConstraint
    destination: MemoryConstraint


@dataclass
class HyperRectangle():
    """
    Represents a multi-dimensional rectangular region in tensor space.

    A HyperRectangle defines a rectangular tile or region within a
    multi-dimensional tensor, specified by its position (offset) and
    dimensions (size) in each axis. This is fundamental for tiled
    processing of tensors where operations are performed on smaller
    rectangular chunks.

    Attributes
    ----------
    offset : Tuple[int, ...]
        Position of the hyperrectangle in feature map space. Each element
        represents the starting index along the corresponding dimension.
    dims : Tuple[int, ...]
        Size of the hyperrectangle along each dimension. Each element
        represents the extent of the rectangle in the corresponding dimension.

    Parameters
    ----------
    offset : Tuple[int, ...]
        Starting position of the rectangle in multi-dimensional space.
    dims : Tuple[int, ...]
        Dimensions/size of the rectangle in multi-dimensional space.

    Raises
    ------
    AssertionError
        If the offset and dims tuples have different lengths.

    Notes
    -----
    The offset and dims must have the same rank (number of dimensions).
    This ensures the hyperrectangle is well-defined in the tensor space.

    Examples
    --------
    >>> rect = HyperRectangle((0, 5), (10, 15))
    >>> # Creates a 2D rectangle starting at (0,5) with size 10x15
    """
    # position of the hyperrectangle in feature map space
    offset: Tuple[int, ...]
    # size of the hyperrectangle
    dims: Tuple[int, ...]

    def __init__(self, offset: Tuple[int, ...], dims: Tuple[int, ...]):
        """
        Initialize a HyperRectangle with given offset and dimensions.

        Parameters
        ----------
        offset : Tuple[int, ...]
            Starting position of the rectangle in multi-dimensional space.
        dims : Tuple[int, ...]
            Dimensions/size of the rectangle in multi-dimensional space.

        Raises
        ------
        AssertionError
            If offset and dims have mismatching dimensions.
        """
        assert len(offset) == len(
            dims), f"HyperRectangle offset and dims for mismatching dimensions {offset} and {dims}"

        self.offset = tuple(offset) if not isinstance(offset, tuple) else offset
        self.dims = tuple(dims) if not isinstance(dims, tuple) else dims


@dataclass
class AbsoluteHyperRectangle:
    """
    Represents a HyperRectangle with an absolute offset in memory space.

    This class combines a HyperRectangle with an absolute memory offset,
    providing both the logical tensor coordinates and the physical memory
    location. This is useful for tracking tiles that have been positioned
    in specific memory locations during tiling operations.

    Attributes
    ----------
    rectangle : HyperRectangle
        The hyperrectangle defining the logical tensor region.
    absoluteOffset : Tuple[int, ...]
        The absolute offset in memory space where this rectangle is located.

    Parameters
    ----------
    rectangle : HyperRectangle
        The hyperrectangle to associate with the absolute offset.
    absoluteOffset : Tuple[int, ...]
        The absolute position in memory space.

    Raises
    ------
    AssertionError
        If the absoluteOffset and rectangle.offset have mismatching dimensions.

    Notes
    -----
    The absoluteOffset must have the same dimensionality as the rectangle's
    offset to ensure consistent coordinate mapping between logical and physical
    memory spaces.
    """
    rectangle: HyperRectangle
    absoluteOffset: Tuple[int, ...]

    def __init__(self, rectangle: HyperRectangle, absoluteOffset: Tuple[int, ...]):
        """
        Initialize an AbsoluteHyperRectangle with rectangle and absolute offset.

        Parameters
        ----------
        rectangle : HyperRectangle
            The hyperrectangle defining the logical tensor region.
        absoluteOffset : Tuple[int, ...]
            The absolute position in memory space.

        Raises
        ------
        AssertionError
            If absoluteOffset and rectangle.offset have mismatching dimensions.
        """
        assert len(absoluteOffset) == len(
            rectangle.offset
        ), f"AsoluteHyperRectangle's absoluteOffset and rectangle's offset for mismatching dimensions {absoluteOffset} and {rectangle.offset}"

        self.rectangle = rectangle
        self.absoluteOffset = absoluteOffset


@dataclass
class TilingSchedule():
    """
    Represents a complete schedule for tiled execution of neural network operations.

    A TilingSchedule defines how data should be loaded, processed, and stored
    during tiled execution. It specifies the memory offsets for input and output
    tensors, as well as the hyperrectangles that define which regions of data
    are processed in each tiling step.

    Attributes
    ----------
    inputBaseOffsets : Dict[str, List[int]]
        Dictionary mapping tensor names to lists of base memory offsets for
        input tiles. Each list should have length equal to the number of tiles.
    outputBaseOffsets : Dict[str, List[int]]
        Dictionary mapping tensor names to lists of base memory offsets for
        output tiles. Each list should have length equal to the number of tiles.
    inputLoadSchedule : List[Dict[str, HyperRectangle]]
        List of dictionaries, one per tile, mapping tensor names to the
        hyperrectangles that should be loaded as input for that tile.
    outputLoadSchedule : List[Dict[str, HyperRectangle]]
        List of dictionaries, one per tile, mapping tensor names to the
        hyperrectangles that should be stored as output for that tile.

    Parameters
    ----------
    inputBaseOffsets : Dict[str, List[int]]
        Input tensor base offsets for each tile.
    outputBaseOffsets : Dict[str, List[int]]
        Output tensor base offsets for each tile.
    inputLoadSchedule : List[Dict[str, HyperRectangle]]
        Input loading schedule for each tile.
    outputLoadSchedule : List[Dict[str, HyperRectangle]]
        Output storing schedule for each tile.

    Notes
    -----
    The lengths of inputLoadSchedule and outputLoadSchedule should typically
    be equal, representing the same number of tiles. Each schedule step
    corresponds to processing one tile of the operation.
    """
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
        """
        Initialize a TilingSchedule with specified offsets and load schedules.

        Parameters
        ----------
        inputBaseOffsets : Dict[str, List[int]]
            Input tensor base offsets for each tile.
        outputBaseOffsets : Dict[str, List[int]]
            Output tensor base offsets for each tile.
        inputLoadSchedule : List[Dict[str, HyperRectangle]]
            Input loading schedule for each tile.
        outputLoadSchedule : List[Dict[str, HyperRectangle]]
            Output storing schedule for each tile.

        Raises
        ------
        AssertionError
            If any key from inputBaseOffsets is missing from a schedule step
            in inputLoadSchedule, or if any key from outputBaseOffsets is
            missing from a schedule step in outputLoadSchedule.
        """

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
        """
        Concatenate two TilingSchedule objects.

        Combines this tiling schedule with another by concatenating their
        load schedules while maintaining the same base offsets. This is
        useful for creating composite tiling schedules from multiple stages.

        Parameters
        ----------
        other : TilingSchedule
            The other TilingSchedule to concatenate with this one.

        Returns
        -------
        TilingSchedule
            A new TilingSchedule containing the concatenated load schedules
            from both input schedules.

        Raises
        ------
        AssertionError
            If the other object is not a TilingSchedule, or if the tensor
            keys don't match between the two schedules.
        """

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
    """
    Defines how variables should be replaced with tile-specific values.

    This class manages the replacement of scalar variables with arrays of
    tile-specific values during tiled execution. It tracks both the per-tile
    replacement values and the corresponding data types for each variable.

    Attributes
    ----------
    perTileReplacements : Dict[str, List]
        Dictionary mapping variable names to lists of replacement values,
        one value per tile. Each list should have length equal to the
        number of tiles.
    replacementTypes : Dict[str, Type[Pointer]]
        Dictionary mapping variable names to their corresponding pointer
        types for the replacement arrays.

    Parameters
    ----------
    perTileReplacements : Dict[str, List]
        Per-tile replacement values for each variable.
    replacementTypes : Dict[str, Type[Pointer]]
        Type information for each replacement variable.

    Raises
    ------
    AssertionError
        If the keys in perTileReplacements and replacementTypes don't match
        exactly, or if they have different numbers of entries.

    Notes
    -----
    This scheme is used to replace compile-time constants with runtime
    arrays during tiled execution, enabling different values for each tile.
    """
    perTileReplacements: Dict[str, List]
    replacementTypes: Dict[str, Type[Pointer]]

    def __init__(self, perTileReplacements: Dict[str, List], replacementTypes: Dict[str, Type[Pointer]]):
        """
        Initialize a VariableReplacementScheme with replacements and types.

        Parameters
        ----------
        perTileReplacements : Dict[str, List]
            Per-tile replacement values for each variable.
        replacementTypes : Dict[str, Type[Pointer]]
            Type information for each replacement variable.

        Raises
        ------
        AssertionError
            If the keys don't match exactly or have different counts.
        """
        assert len(perTileReplacements.keys()) == len(
            replacementTypes.keys()), "Exactly all replacements must have one type"

        for key in perTileReplacements.keys():
            assert key in replacementTypes.keys(), "Keys must match!"

        self.perTileReplacements = perTileReplacements
        self.replacementTypes = replacementTypes

    def __add__(self, other: VariableReplacementScheme) -> VariableReplacementScheme:
        """
        Concatenate two VariableReplacementScheme objects.

        Combines this replacement scheme with another by concatenating their
        per-tile replacement lists. This is useful for merging replacement
        schemes from multiple tiling stages.

        Parameters
        ----------
        other : VariableReplacementScheme
            The other VariableReplacementScheme to concatenate with this one.

        Returns
        -------
        VariableReplacementScheme
            A new VariableReplacementScheme with concatenated replacement lists.

        Raises
        ------
        AssertionError
            If the other object is not a VariableReplacementScheme, or if
            the variable keys don't match between the two schemes.
        """

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
    """
    Optimize a variable replacement scheme by eliminating constant replacements.

    Analyzes the replacement scheme and removes variables that have the same
    value across all tiles, directly setting them in the operator representation
    instead. This optimization reduces memory usage and improves performance.

    Parameters
    ----------
    scheme : VariableReplacementScheme
        The original variable replacement scheme to optimize.
    operatorRepresentation : OperatorRepresentation
        The operator representation that will be updated with constant values.

    Returns
    -------
    Tuple[VariableReplacementScheme, Dict]
        A tuple containing:
        - The minimized VariableReplacementScheme with only non-constant variables
        - A dictionary of updates to apply to the operator representation

    Notes
    -----
    Variables with identical values across all tiles are considered constants
    and are removed from the replacement scheme. Their single value is set
    directly in the operator representation.
    """
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
    """
    Minimize a hyperrectangle by collapsing dimensions where possible.

    Reduces the dimensionality of a hyperrectangle by merging consecutive
    dimensions where the rectangle spans the entire reference shape. This
    optimization is useful for memory transfers and reduces complexity.

    Parameters
    ----------
    rect : HyperRectangle
        The hyperrectangle to minimize.
    referenceShape : Sequence[int]
        The shape of the reference tensor that the rectangle is within.

    Returns
    -------
    Tuple[HyperRectangle, Tuple[int, ...]]
        A tuple containing:
        - The minimized HyperRectangle with collapsed dimensions
        - The corresponding minimized reference shape

    Raises
    ------
    AssertionError
        If the rectangle offset is non-zero when dimensions match the
        reference shape (indicating the rectangle spans the full dimension).

    Notes
    -----
    Dimensions are collapsed from right to left. When a rectangle dimension
    equals the reference dimension and has zero offset, it can be merged
    with adjacent dimensions to reduce the overall rank.

    Example
    -------
    >>> rect = HyperRectangle((0, 0), (2, 2))
    >>> minimizeRectangle(rect, (4, 4))
        (HyperRectangle(offset=(0, 0), dims=(2, 2)), (2, 4))
    >>> rect = HyperRectangle((0, 0), (2, 2))
    >>> minimizeRectangle(rect, (4, 2))
        (HyperRectangle(offset=(0,), dims=(4,)), (8,))
    """
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
    """
    Pad a shape tuple to a target rank by prepending ones.

    Extends a shape tuple to a higher dimensionality by adding leading
    dimensions of size 1. This is useful for broadcasting operations
    and ensuring consistent tensor ranks.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The original shape tuple to pad.
    rank : int
        The target rank (number of dimensions) for the padded shape.

    Returns
    -------
    Tuple[int, ...]
        The padded shape tuple with leading dimensions of size 1.

    Raises
    ------
    AssertionError
        If the target rank is smaller than the current shape's rank.

    Examples
    --------
    >>> padShape((3, 4), 4)
    (1, 1, 3, 4)
    >>> padShape((5,), 3)
    (1, 1, 5)
    """
    assert rank >= len(
        shape), f"Cannot pad to rank smaller then shape's. Received rank: {rank}, shape rank: {len(shape)}"
    ret = tuple([1] * (rank - len(shape))) + shape
    assert len(ret) == rank
    return ret


def padOffset(offset: Tuple[int, ...], rank: int) -> Tuple[int, ...]:
    """
    Pad an offset tuple to a target rank by prepending zeros.

    Extends an offset tuple to a higher dimensionality by adding leading
    offset values of 0. This ensures offset tuples match the rank of
    their corresponding shapes.

    Parameters
    ----------
    offset : Tuple[int, ...]
        The original offset tuple to pad.
    rank : int
        The target rank (number of dimensions) for the padded offset.

    Returns
    -------
    Tuple[int, ...]
        The padded offset tuple with leading zeros.

    Raises
    ------
    AssertionError
        If the target rank is smaller than the current offset's rank.

    Examples
    --------
    >>> padOffset((2, 3), 4)
    (0, 0, 2, 3)
    >>> padOffset((5,), 3)
    (0, 0, 5)
    """
    assert rank >= len(
        offset), f"Cannot pad to rank smaller then offset's. Received rank: {rank}, offset rank: {len(offset)}"
    ret = tuple([0] * (rank - len(offset))) + offset
    assert len(ret) == rank
    return ret


def padStride(stride: Tuple[int, ...], rank: int, paddingStride: int) -> Tuple[int, ...]:
    """
    Pad a stride tuple to a target rank by prepending a specified stride value.

    Extends a stride tuple to a higher dimensionality by adding leading
    stride values. This is useful for maintaining consistent stride
    calculations across different tensor ranks.

    Parameters
    ----------
    stride : Tuple[int, ...]
        The original stride tuple to pad.
    rank : int
        The target rank (number of dimensions) for the padded stride.
    paddingStride : int
        The stride value to use for padding (prepended dimensions).

    Returns
    -------
    Tuple[int, ...]
        The padded stride tuple with leading padding stride values.

    Raises
    ------
    AssertionError
        If the target rank is smaller than the current stride's rank.

    Examples
    --------
    >>> padStride((4, 1), 4, 16)
    (16, 16, 4, 1)
    >>> padStride((1,), 3, 8)
    (8, 8, 1)
    """
    assert rank >= len(
        stride), f"Cannot pad to rank smaller then stride's. Received rank: {rank}, stride rank: {len(stride)}"
    ret = tuple([paddingStride] * (rank - len(stride))) + stride
    assert len(ret) == rank
    return ret


def stridesFromShape(shape: Sequence[int]) -> Tuple[int, ...]:
    """
    Calculate memory strides from a tensor shape.

    Computes the stride values for each dimension of a tensor based on its
    shape. Strides represent the number of elements to skip in memory when
    moving one position along each dimension.

    Parameters
    ----------
    shape : Sequence[int]
        The shape of the tensor as a sequence of dimension sizes.

    Returns
    -------
    Tuple[int, ...]
        The stride values for each dimension, where the last dimension
        has stride 1 and earlier dimensions have progressively larger strides.

    Notes
    -----
    Strides are computed assuming row-major (C-style) memory layout.
    The stride for dimension i is the product of all dimensions after i.

    Examples
    --------
    >>> stridesFromShape([2, 3, 4])
    (12, 4, 1)
    >>> stridesFromShape([5, 6])
    (6, 1)
    """
    strides = [1] * len(shape)
    for idx, dim in enumerate(reversed(shape[1:])):
        strides[idx + 1] = strides[idx] * dim
    return tuple(reversed(strides))


def calculateFlatOffset(offsets: Sequence[int], strides: Sequence[int]) -> int:
    """
    Calculate the flat memory offset from multi-dimensional coordinates.

    Converts multi-dimensional tensor coordinates (offsets) to a single
    flat memory offset using the provided stride information. This is
    essential for translating tensor indices to memory addresses.

    Parameters
    ----------
    offsets : Sequence[int]
        The multi-dimensional coordinates/offsets in each dimension.
    strides : Sequence[int]
        The stride values for each dimension.

    Returns
    -------
    int
        The flat memory offset corresponding to the multi-dimensional position.

    Raises
    ------
    AssertionError
        If offsets and strides have different numbers of dimensions.

    Notes
    -----
    The flat offset is computed as the sum of (offset[i] * stride[i])
    for all dimensions i.

    Examples
    --------
    >>> calculateFlatOffset([1, 2, 3], [12, 4, 1])
    23
    >>> calculateFlatOffset([0, 1], [6, 1])
    1
    """
    assert len(offsets) == len(strides), \
        f"Offsets and strides have to have the same number of dimensions. Length offsets: {len(offsets)}, strides: {len(strides)}"
    return sum(offset * stride for offset, stride in zip(offsets, strides))


def calculateFlatOffsetInBytes(tile: HyperRectangle, referenceBuffer: VariableBuffer) -> int:
    """
    Calculate the flat memory offset in bytes for a hyperrectangle tile.

    Computes the byte offset in memory for the starting position of a
    hyperrectangle tile within a reference buffer. This accounts for
    both the multi-dimensional positioning and the data type size.

    Parameters
    ----------
    tile : HyperRectangle
        The hyperrectangle tile whose offset should be calculated.
    referenceBuffer : VariableBuffer
        The reference buffer containing the tile, used for shape and type info.

    Returns
    -------
    int
        The flat memory offset in bytes from the buffer start to the tile start.

    Notes
    -----
    The calculation combines multi-dimensional offset computation with
    data type width to produce a byte-level memory offset.
    """
    return int(
        calculateFlatOffset(tile.offset, stridesFromShape(referenceBuffer.shape)) *
        (referenceBuffer._type.referencedType.typeWidth // 8))


def computeTileHyperRectangles(memoryTransfer: MemoryTransfer) -> List[HyperRectangle]:
    """
    Compute hyperrectangle tiles for a memory transfer operation.

    Generates a list of hyperrectangle tiles that partition the source tensor
    into smaller chunks that fit within the destination memory constraints.
    This is fundamental for tiled execution where large tensors are processed
    in smaller, memory-efficient pieces.

    Parameters
    ----------
    memoryTransfer : MemoryTransfer
        The memory transfer operation defining source and destination constraints.

    Returns
    -------
    List[HyperRectangle]
        A list of hyperrectangle tiles that cover the entire source tensor,
        each fitting within the destination memory constraints.

    Raises
    ------
    AssertionError
        If source or destination shapes are undefined, if they have different
        numbers of dimensions, or if any destination dimension is larger than
        the corresponding source dimension.

    Notes
    -----
    The tiling algorithm generates non-overlapping tiles that completely
    cover the source tensor. Each tile is sized to fit within the destination
    memory constraints, with edge tiles potentially being smaller to fit
    exactly within the source tensor boundaries.

    The tiles are generated in row-major order, iterating through dimensions
    from outermost to innermost.
    """
    assert memoryTransfer.source.shape is not None, "Source transfer shape cannot be undefined!"
    assert memoryTransfer.destination.shape is not None, "Destination transfer shape cannot be undefined!"

    assert len(memoryTransfer.source.shape) == len(memoryTransfer.destination.shape), \
    f"Source and target of memory transfer {memoryTransfer} don't have the same number of dimensions!"

    largeShape = memoryTransfer.source.shape
    smallShape = memoryTransfer.destination.shape

    for dimIdx, (dimSizeSmall, dimSizeLarge) in enumerate(zip(smallShape, largeShape)):
        assert dimSizeSmall <= dimSizeLarge, f"smallShape[{dimIdx}] should not be bigger then largeShape[{dimIdx}]. ({dimSizeSmall} > {dimSizeLarge})"

    def nextTileIndex(tileIndexEnd: List[int]) -> Generator[List[int]]:
        """
        Generate tile indices in row-major order.

        Parameters
        ----------
        tileIndexEnd : List[int]
            The end index for each dimension (exclusive).

        Yields
        ------
        List[int]
            Successive tile indices covering the entire index space.
        """
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
