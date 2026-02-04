# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext
from Deeploy.DeeployTypes import OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle
from Deeploy.TilingExtension.TilingCodegen import TilingSchedule
from Deeploy.TilingExtension.TilingCodegen import VariableReplacementScheme


class ReshapeTileConstraint(TileConstraint):
    """Tile constraint for Reshape operation - a NOP that just reinterprets data layout."""

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        pointer: List[str] = []

        for key, value in parseDict.items():
            if not isinstance(value, str):
                continue

            if ctxt.is_global(value) or ctxt.is_local(value):
                pointer.append(value)

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            _buffer = ctxt.lookup(bufferName)
            tilerModel.addTensorDimToModel(ctxt, bufferName)

            for idx, shapeDim in enumerate(_buffer.shape):
                tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = bufferName, dimIdx = idx) <= shapeDim)

        # Constrain total elements to be equal
        inputBuffer = ctxt.lookup(inputBufferName)
        outputBuffer = ctxt.lookup(outputBufferName)

        # For reshape, we want the tiles to have the same total number of elements
        # This is automatically satisfied if we tile based on output and compute input from that

        # Remove unused tensors from deployment
        for bufferName in pointer:
            if bufferName not in [inputBufferName, outputBufferName]:
                ctxt.lookup(bufferName)._deploy = False

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # For reshape, input and output have the same data, just different interpretations
        # We need to compute the corresponding input cube for each output cube
        inputName = operatorRepresentation['data_in']
        outputName = operatorRepresentation['data_out']
        inputShape = ctxt.lookup(inputName).shape
        outputShape = ctxt.lookup(outputName).shape

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            # Calculate the flat offset and size for the output cube
            outSize = np.prod(cube.dims)
            replacements["size"].append(outSize)

            # For reshape, we need to map output cube to input cube
            # Calculate flat index range for output cube
            outOffset = 0
            outStrides = []
            stride = 1
            for dim in reversed(outputShape):
                outStrides.insert(0, stride)
                stride *= dim

            for i, (off, dim) in enumerate(zip(cube.offset, cube.dims)):
                outOffset += off * outStrides[i]

            # Convert flat offset to input coordinates
            inStrides = []
            stride = 1
            for dim in reversed(inputShape):
                inStrides.insert(0, stride)
                stride *= dim

            inOffset = []
            remaining = outOffset
            for i, stride in enumerate(inStrides):
                inOffset.append(remaining // stride)
                remaining = remaining % stride

            # Calculate input cube dimensions
            # For simplicity, treat as 1D cube in input space
            inCubeDims = list(inputShape)
            inCubeOffset = [0] * len(inputShape)

            # Set the last dimension to the size, and offset based on flat index
            totalSize = outSize
            if len(inputShape) > 0:
                # Compute proper input cube that covers the same elements
                # Use a simple approach: linearize the input
                inCubeOffset = list(inOffset)
                inCubeDims = [1] * len(inputShape)
                inCubeDims[-1] = min(totalSize, inputShape[-1] - inCubeOffset[-1])
                remaining = totalSize - inCubeDims[-1]

                for i in range(len(inputShape) - 2, -1, -1):
                    if remaining <= 0:
                        break
                    inCubeDims[i] = min(remaining // np.prod(inputShape[i + 1:]) + 1, inputShape[i])
                    remaining -= (inCubeDims[i] - 1) * np.prod(inputShape[i + 1:])

            inputCube = HyperRectangle(tuple(inCubeOffset), tuple(inCubeDims))
            inputLoadSchedule.append({"data_in": inputCube})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
