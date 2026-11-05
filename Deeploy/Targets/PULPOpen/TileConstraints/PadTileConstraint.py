# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class Pad2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        channels_first = bool(parseDict.get('channels_first', 1))

        if channels_first:
            batchDim = 0
            channelDim = 1
            heightDim = 2
            widthDim = 3
        else:
            batchDim = 0
            heightDim = 1
            widthDim = 2
            channelDim = 3

        inputBatchVar = tilerModel.getTensorDimVar(inputBufferName, batchDim)
        inputHeightVar = tilerModel.getTensorDimVar(inputBufferName, heightDim)
        inputWidthVar = tilerModel.getTensorDimVar(inputBufferName, widthDim)
        inputChannelVar = tilerModel.getTensorDimVar(inputBufferName, channelDim)

        outputBatchVar = tilerModel.getTensorDimVar(outputBufferName, batchDim)
        outputHeightVar = tilerModel.getTensorDimVar(outputBufferName, heightDim)
        outputWidthVar = tilerModel.getTensorDimVar(outputBufferName, widthDim)
        outputChannelVar = tilerModel.getTensorDimVar(outputBufferName, channelDim)

        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputWidthVar == inputWidthVar + 2 * parseDict['pad_x'])
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)

        # The input buffer must hold the input rows of every output tile. A tile of height T can lie entirely in
        # the data region and then needs T rows, so only the full-tensor bound H_in (the upper bound of the variable)
        # allows fewer: inputHeight == min(T, H_in). This disjunction says inputHeight >= T or inputHeight == H_in;
        # the second constraint caps it at T.
        inputHeightTotal = ctxt.lookup(inputBufferName).shape[heightDim]
        tilerModel.addConstraint((inputHeightVar >= outputHeightVar) + (inputHeightVar == inputHeightTotal) >= 1)
        tilerModel.addConstraint(outputHeightVar >= inputHeightVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        channels_first = bool(parseDict.get('channels_first', 1))

        if channels_first:
            batchDim = 0
            channelDim = 1
            heightDim = 2
            widthDim = 3
        else:
            batchDim = 0
            heightDim = 1
            widthDim = 2
            channelDim = 3

        inputBuffer = ctxt.lookup(inputBufferName)
        outputBuffer = ctxt.lookup(outputBufferName)

        # Keep batch, width and channels whole to preserve contiguous NHWC row copies.
        tilerModel.addConstraint(tilerModel.getTensorDimVar(inputBufferName, batchDim) == inputBuffer.shape[batchDim])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(outputBufferName, batchDim) == outputBuffer.shape[batchDim])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(inputBufferName, widthDim) == inputBuffer.shape[widthDim])
        tilerModel.addConstraint(tilerModel.getTensorDimVar(outputBufferName, widthDim) == outputBuffer.shape[widthDim])
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(inputBufferName, channelDim) == inputBuffer.shape[channelDim])
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(outputBufferName, channelDim) == outputBuffer.shape[channelDim])

        # A tile taller than pad_y always contains input data unless it is the remainder (tail) tile of its parent:
        # the first tile covers the pad_y top rows plus at least one data row, and a full-size tile that started in
        # the bottom padding would end beyond the output. A tail tile made only of bottom padding is handled in
        # _computeInputCube.
        outputHeightVar = tilerModel.getTensorDimVar(outputBufferName, heightDim)
        tilerModel.addConstraint(outputHeightVar >= (parseDict['pad_y'] + 1))

        return tilerModel

    @staticmethod
    def _computeInputCube(outputCube: HyperRectangle, parseDict: Dict,
                          inputShape: Tuple[int, ...]) -> Tuple[HyperRectangle, int, int]:
        """Return the input cube of ``outputCube`` and its tile-local top and bottom padding.

        ``outputCube.offset`` must be in full-output coordinates, and the returned cube is in full-input
        coordinates. The input sizes come from ``inputShape`` (the full input tensor), not from ``parseDict``:
        ``TilingVariableReplacementUpdate`` serializes again after ``dim_im_in_*`` have been replaced by per-tile
        reference names.

        The local padding is clamped so that ``localPadTop + localPadBottom <= outputHSize``; the kernel copies
        ``outputHSize - localPadTop - localPadBottom`` input rows.

        A tile that contains no input row (only the remainder tile of a parent can, and only in the bottom padding)
        gets ``localPadTop + localPadBottom == outputHSize``, so the kernel copies nothing and only fills the pad
        value. The DMA still needs a non-empty transfer, so its input cube is one row clamped into
        ``[0, dim_im_in_x)`` (the last input row for bottom padding). That row is never read by the kernel and fits
        the input buffer, whose height variable has a lower bound of 1.
        """
        channels_first = bool(parseDict.get('channels_first', 1))

        if channels_first:
            batchOffset, _, outputHOffset, _ = outputCube.offset
            batchSize, _, outputHSize, _ = outputCube.dims
            _, inputChannels, inputHeightTotal, inputWidth = inputShape
        else:
            batchOffset, outputHOffset, _, _ = outputCube.offset
            batchSize, outputHSize, _, _ = outputCube.dims
            _, inputHeightTotal, inputWidth, inputChannels = inputShape

        padTop = parseDict['pad_y']

        dataStart = padTop
        dataEnd = padTop + inputHeightTotal
        tileEnd = outputHOffset + outputHSize

        localPadTop = min(max(dataStart - outputHOffset, 0), outputHSize)
        localPadBottom = min(max(tileEnd - dataEnd, 0), outputHSize - localPadTop)

        inputStart = max(outputHOffset, dataStart) - dataStart
        inputEnd = min(tileEnd, dataEnd) - dataStart
        if inputEnd > inputStart:
            inputHeight = inputEnd - inputStart
        else:
            # Padding-only tile: load one in-bounds dummy row instead of the row past the end of the input.
            inputStart = min(inputStart, inputHeightTotal - 1)
            inputHeight = 1

        if channels_first:
            inCube = HyperRectangle((batchOffset, 0, inputStart, 0),
                                    (batchSize, inputChannels, inputHeight, inputWidth))
        else:
            inCube = HyperRectangle((batchOffset, inputStart, 0, 0),
                                    (batchSize, inputHeight, inputWidth, inputChannels))

        return inCube, localPadTop, localPadBottom

    @staticmethod
    def _inputLevelIndex(tilingSolution: NodeMemoryConstraint, targetMemLevel: str,
                         operatorRepresentation: OperatorRepresentation) -> int:
        """Position of ``targetMemLevel`` in the memory path of ``data_in`` (-1 if absent).

        Index 1 means ``data_in`` tiles are cut from the full tensor (first tiling level), index >= 2 means they
        are cut from an input tile of the enclosing tiling level, and 0/-1 mean the input is not transferred into
        ``targetMemLevel`` and is read in place. Same convention as ``SliceTileConstraint.inputLevelIndex``.
        """
        inputPath = list(
            tilingSolution.tensorMemoryConstraints[operatorRepresentation['data_in']].memoryConstraints.keys())
        return inputPath.index(targetMemLevel) if targetMemLevel in inputPath else -1

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        """Serialize the output tiles of one parent tile.

        ``wrapTilingSolution`` calls this once per parent output tile with that parent's children.
        ``cube.rectangle.offset`` is relative to the parent output tile, ``cube.absoluteOffset`` to the full output
        tensor. The padding only depends on where a tile lies in the full output, so the tile-local padding and the
        input rows are always computed from the absolute offset (:meth:`_computeInputCube`). The input cube is then
        expressed relative to the parent ``data_in`` buffer:

        * First tiling level (the parent buffer is the full input tensor): the global cube is used as is.
        * Nested tiling (e.g. L3 -> L2 -> L1): the parent buffer holds the global input cube of the parent output
          tile, produced by this method one level up. The child cube is shifted by that cube's offset; it lies
          inside it because the input rows of a sub-range of the parent are a sub-range of the parent's rows (a
          padding-only child of a padding-only parent maps onto the same clamped dummy row).
        * ``data_in`` is not transferred to ``targetMemLevel`` (e.g. ``TilingVariableReplacement("L2")`` when L2 is
          the home level, where ``wrapTilingSolution`` serializes the L1 tiles): the global cube is kept, as at the
          first level.
        """
        inputLevelIdx = cls._inputLevelIndex(tilingSolution, targetMemLevel, operatorRepresentation)
        fullInputShape = tuple(ctxt.lookup(operatorRepresentation['data_in']).shape)

        parentInputOffset = None
        if inputLevelIdx >= 2:
            firstCube = absoluteOutputCubes[0]
            parentOffset = tuple(a - r for a, r in zip(firstCube.absoluteOffset, firstCube.rectangle.offset))
            # The children partition the parent (computeTileHyperRectangles), so this is its extent.
            parentDims = tuple(
                max(cube.rectangle.offset[axis] + cube.rectangle.dims[axis]
                    for cube in absoluteOutputCubes)
                for axis in range(len(parentOffset)))
            parentInCube, _, _ = cls._computeInputCube(HyperRectangle(parentOffset, parentDims), operatorRepresentation,
                                                       fullInputShape)
            parentInputOffset, parentInputDims = parentInCube.offset, parentInCube.dims

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements: Dict[str, List[int]] = {
            "batch": [],
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_in_ch": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "dim_im_out_ch": [],
            "pad_y_top": [],
            "pad_y_bottom": [],
            "data_out_size": [],
        }

        replacementTypes = {
            "batch": PointerClass(uint16_t),
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_in_ch": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "dim_im_out_ch": PointerClass(uint16_t),
            "pad_y_top": PointerClass(uint16_t),
            "pad_y_bottom": PointerClass(uint16_t),
            # Elements per output tile; can exceed 65535 (e.g. int8 tiles in a large L1).
            "data_out_size": PointerClass(uint32_t),
        }

        channels_first = bool(operatorRepresentation.get('channels_first', 1))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for absoluteCube in absoluteOutputCubes:
            outCube = absoluteCube.rectangle
            globalOutCube = HyperRectangle(tuple(absoluteCube.absoluteOffset), tuple(outCube.dims))
            inCube, localPadTop, localPadBottom = cls._computeInputCube(globalOutCube, operatorRepresentation,
                                                                        fullInputShape)

            if parentInputOffset is not None:
                localOffset = tuple(o - p for o, p in zip(inCube.offset, parentInputOffset))
                assert all(0 <= o and o + d <= p for o, d, p in zip(localOffset, inCube.dims, parentInputDims)), \
                    f"Pad2D input tile {inCube} exceeds its parent input tile {parentInCube}"
                inCube = HyperRectangle(localOffset, inCube.dims)

            replacements["batch"].append(outCube.dims[0])
            replacements["pad_y_top"].append(localPadTop)
            replacements["pad_y_bottom"].append(localPadBottom)
            replacements["data_out_size"].append(int(np.prod(outCube.dims)))

            if channels_first:
                replacements["dim_im_in_ch"].append(inCube.dims[1])
                replacements["dim_im_in_x"].append(inCube.dims[2])
                replacements["dim_im_in_y"].append(inCube.dims[3])
                replacements["dim_im_out_ch"].append(outCube.dims[1])
                replacements["dim_im_out_x"].append(outCube.dims[2])
                replacements["dim_im_out_y"].append(outCube.dims[3])
            else:
                replacements["dim_im_in_x"].append(inCube.dims[1])
                replacements["dim_im_in_y"].append(inCube.dims[2])
                replacements["dim_im_in_ch"].append(inCube.dims[3])
                replacements["dim_im_out_x"].append(outCube.dims[1])
                replacements["dim_im_out_y"].append(outCube.dims[2])
                replacements["dim_im_out_ch"].append(outCube.dims[3])

            inputLoadSchedule.append({"data_in": inCube})
            outputLoadSchedule.append({"data_out": outCube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
