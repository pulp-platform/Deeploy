# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class GlobalAveragePoolTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Get the input buffer to determine dimensionality
        inputBuffer = ctxt.lookup(inputBufferName)
        numDims = len(inputBuffer.shape)

        if numDims == 4:  # 2D case: [N, C, H, W]
            # Batch dimension: input and output must be equal
            inputBatchVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=0)
            outputBatchVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=0)
            tilerModel.addConstraint(outputBatchVar == inputBatchVar)

            # Channel dimension: input and output must be equal
            inputChannelVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=1)
            outputChannelVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=1)
            tilerModel.addConstraint(outputChannelVar == inputChannelVar)

            # Spatial dimensions: output is always 1x1 (reduced)
            outputHeightVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=2)
            outputWidthVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=3)
            tilerModel.addConstraint(outputHeightVar == 1)
            tilerModel.addConstraint(outputWidthVar == 1)

        elif numDims == 3:  # 1D case: [N, C, L]
            # Batch dimension: input and output must be equal
            inputBatchVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=0)
            outputBatchVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=0)
            tilerModel.addConstraint(outputBatchVar == inputBatchVar)

            # Channel dimension: input and output must be equal
            inputChannelVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=1)
            outputChannelVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=1)
            tilerModel.addConstraint(outputChannelVar == inputChannelVar)

            # Length dimension: output is always 1 (reduced)
            outputLengthVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=2)
            tilerModel.addConstraint(outputLengthVar == 1)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Add tiling policy constraints. For GlobalAveragePool, we can tile along batch and channel dimensions,
        but spatial dimensions must remain full to compute the average correctly.
        """
        inputBufferName = parseDict['data_in']
        inputBuffer = ctxt.lookup(inputBufferName)
        numDims = len(inputBuffer.shape)

        if numDims == 4:  # 2D case: [N, C, H, W]
            # Spatial dimensions must be full (no tiling on H, W)
            inputHeightVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=2)
            inputWidthVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=3)
            tilerModel.addConstraint(inputHeightVar == inputBuffer.shape[2])
            tilerModel.addConstraint(inputWidthVar == inputBuffer.shape[3])

        elif numDims == 3:  # 1D case: [N, C, L]
            # Length dimension must be full (no tiling on L)
            inputLengthVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=2)
            tilerModel.addConstraint(inputLengthVar == inputBuffer.shape[2])

        return tilerModel

    @staticmethod
    def computeInputCubeFromOutputCube(outputCube: HyperRectangle, parseDict: Dict,
                                       inputShape: Tuple[int, ...]) -> HyperRectangle:
        """
        Compute the input cube required for a given output cube.
        Since spatial dimensions are reduced, the input cube needs full spatial dimensions.
        """
        numDims = len(inputShape)

        if numDims == 4:  # 2D case: [N, C, H, W]
            # Input cube: same batch and channel as output, but full spatial dimensions
            in_cube_offset = (outputCube.offset[0], outputCube.offset[1], 0, 0)
            in_cube_dims = (outputCube.dims[0], outputCube.dims[1], inputShape[2], inputShape[3])
        elif numDims == 3:  # 1D case: [N, C, L]
            # Input cube: same batch and channel as output, but full length dimension
            in_cube_offset = (outputCube.offset[0], outputCube.offset[1], 0)
            in_cube_dims = (outputCube.dims[0], outputCube.dims[1], inputShape[2])
        else:
            raise ValueError(f"Unsupported number of dimensions: {numDims}")

        return HyperRectangle(offset=in_cube_offset, dims=in_cube_dims)

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        """
        Serialize the tiling solution into schedules and variable replacements.
        """
        # Extract base addresses
        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Get input shape from context
        inputBuffer = ctxt.lookup(operatorRepresentation['data_in'])
        inputShape = inputBuffer.shape
        numDims = len(inputShape)

        # Prepare replacement dictionaries
        replacements: Dict[str, List[int]] = {
            "batch": [],
            "channels": [],
            "ch_im_in": [],
            "size": [],
        }

        if numDims == 4:  # 2D case
            replacements["dim_im_in_x"] = []
            replacements["dim_im_in_y"] = []
            replacementTypes = {
                "batch": PointerClass(uint16_t),
                "channels": PointerClass(uint16_t),
                "ch_im_in": PointerClass(uint16_t),
                "dim_im_in_x": PointerClass(uint16_t),
                "dim_im_in_y": PointerClass(uint16_t),
                "size": PointerClass(uint16_t),
            }
        else:  # 1D case
            replacements["dim_im_in_y"] = []
            replacementTypes = {
                "batch": PointerClass(uint16_t),
                "channels": PointerClass(uint16_t),
                "ch_im_in": PointerClass(uint16_t),
                "dim_im_in_y": PointerClass(uint16_t),
                "size": PointerClass(uint16_t),
            }

        # Prepare loading schedules
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Iterate over output cubes
        for out_cube in [cube.rectangle for cube in absoluteOutputCubes]:
            # Compute corresponding input cube
            in_cube = cls.computeInputCubeFromOutputCube(out_cube, parseDict=operatorRepresentation,
                                                        inputShape=inputShape)

            # Extract dimensions from cubes
            if numDims == 4:  # 2D case
                batch_size = in_cube.dims[0]
                channel_size = in_cube.dims[1]
                height_size = in_cube.dims[2]
                width_size = in_cube.dims[3]
                size = int(height_size * width_size)

                replacements["batch"].append(batch_size)
                replacements["channels"].append(channel_size)
                replacements["ch_im_in"].append(channel_size)
                replacements["dim_im_in_x"].append(height_size)
                replacements["dim_im_in_y"].append(width_size)
                replacements["size"].append(size)

            else:  # 1D case
                batch_size = in_cube.dims[0]
                channel_size = in_cube.dims[1]
                length_size = in_cube.dims[2]
                size = int(length_size)

                replacements["batch"].append(batch_size)
                replacements["channels"].append(channel_size)
                replacements["ch_im_in"].append(channel_size)
                replacements["dim_im_in_y"].append(length_size)
                replacements["size"].append(size)

            # Append to schedules
            inputLoadSchedule.append({"data_in": in_cube})
            outputLoadSchedule.append({"data_out": out_cube})

        # Create schedule and replacement scheme
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
