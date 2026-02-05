# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class SliceTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # ===== GET NECESSARY INFORMATION =====
        #   Get I/O buffer names
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        #   Get I/O shapes
        inputShape = parseDict['data_in_shape']

        #   Get other necessary information
        sliceAxes = parseDict['axes']
        sliceSteps = parseDict['steps']

        # ===== ADD I/O DIMENSIONS TO THE MODEL AS VARIABLES =====
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # ===== ADD CONSTRAINTS =====
        #   Add constraints for the I/O dimensions
        for idx in range(len(inputShape)):
            # Get current dimension variables
            inputDimensionVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx)
            outputDimensionVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)

            if idx in sliceAxes:
                # For sliced axes, constrain to minimal input dimension
                # based on the output dimension and the slicing step
                axIndex = list(sliceAxes).index(idx)
                axStep = sliceSteps[axIndex]

                tilerModel.addConstraint(inputDimensionVar == ((outputDimensionVar - 1) * axStep + 1))
            else:
                # Otherwise, input and output dimensions need to be equal
                tilerModel.addConstraint(outputDimensionVar == inputDimensionVar)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        symbolicParseDict = parseDict.copy()

        return symbolicParseDict

    @staticmethod
    def computeInputCubeFromOutputCube(outputCube: AbsoluteHyperRectangle, parseDict: Dict) -> HyperRectangle:
        # Computes the input cube given the output cube and the slicing parameters.
        #
        # Will provide a minimal input cube, that only requires the data needed for the output cube
        # by ignoring the input data that is outside of the slicing scope,
        # as given by the slicing starting and ending parameters.
        #
        # (It will start with the first element required for the output cube,
        # and will end with the last element required for the output cube).
        #
        # *Function is ready for multiple axes slicing.

        # Start from the output cube dimensions and offsets
        in_cube_dims = list(outputCube.dims).copy()
        in_cube_offset = list(outputCube.offset).copy()

        # Iterate through the sliced axes
        for idx, ax in enumerate(parseDict['axes']):
            # Get current sliced ax parameters
            start = parseDict['starts'][idx]
            step = parseDict['steps'][idx]

            # Compute input cube parameters for the current axis
            in_cube_dims[ax] = (outputCube.dims[ax] - 1) * step + 1
            in_cube_offset[ax] = start + outputCube.offset[ax] * step

        return HyperRectangle(offset = tuple(in_cube_offset), dims = tuple(in_cube_dims))

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        # Extract rectangle information (offsets and dimensions) from output cubes
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        # Prepare address names
        addrNames = ['data_in', 'data_out']

        # Extract memory base addresses for each of the required components,
        # based on the computed memory configuration
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Prepare replacement lists for the elements inside the operator representation,
        # for the cubes to be computed further down in this function

        # Build replacementTypes based on the actual number of dimensions
        data_in_shape_type = [PointerClass(uint16_t) for _ in range(len(operatorRepresentation['data_in_shape']))]
        data_out_shape_type = [PointerClass(uint16_t) for _ in range(len(operatorRepresentation['data_out_shape']))]

        replacements = {
            "data_in_shape": [],
            "data_out_shape": [],
            "starts": [],
            "ends": [],
            "data_in_size": [],
        }

        replacementTypes = {
            "data_in_shape": data_in_shape_type,
            "data_out_shape": data_out_shape_type,
            "starts": PointerClass(uint16_t),
            "ends": PointerClass(uint16_t),
            "data_in_size": PointerClass(uint16_t),
        }

        # Prepare loading schedule lists
        inputLoadSchedule = []
        outputLoadSchedule = []

        for out_cube in outputCubes:
            # Compute input cube
            in_cube = SliceTileConstraint.computeInputCubeFromOutputCube(out_cube, parseDict = operatorRepresentation)

            # Compute new starts and ends for replacement
            new_starts = list()
            new_ends = list()
            for ax in operatorRepresentation['axes']:
                new_starts.append(in_cube.offset[ax])
                new_ends.append(in_cube.offset[ax] + in_cube.dims[ax])

            # Append replacement elements (using tuples so they can be hashed by minimizeVariableReplacement)
            replacements["data_in_shape"].append(tuple(in_cube.dims))
            replacements["data_out_shape"].append(tuple(out_cube.dims))
            replacements["starts"].append(tuple(new_starts))
            replacements["ends"].append(tuple(new_ends))
            replacements["data_in_size"].append(int(np.prod(in_cube.dims)))

            # Append new cubes
            inputLoadSchedule.append({"data_in": in_cube})
            outputLoadSchedule.append({"data_out": out_cube})

        # Prepare containing objects with information computed in this function regarding tiling schedule
        # and variable replacement inside operator representation
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
