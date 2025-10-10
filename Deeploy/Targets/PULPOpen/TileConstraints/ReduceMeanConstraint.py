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


class ReduceMeanTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Get necessary information
        #   Get I/O buffer names
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        #   Get I/O shapes
        outputShape = parseDict['data_out_shape']

        #   Get other necessary information
        reduceAxes = parseDict['axes']
        keepDims = parseDict['keepdims']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Add constratints for the I/O dimensions
        input_ax = 0
        for idx in range(len(outputShape)):
            # Get current dimension variables
            outputDimensionVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)

            if idx in reduceAxes:
                # For reduced axes, constrain to 1 if keepdims is set,
                # otherwise skip this axis in the input tensor,
                # as it needs to be eliminated.
                if keepDims:
                    tilerModel.addConstraint(outputDimensionVar == 1)
                    input_ax += 1
            else:
                # Otherwise, input and output dimensions need to be equal
                inputDimensionVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = input_ax)

                tilerModel.addConstraint(outputDimensionVar == inputDimensionVar)

                input_ax += 1

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # TODO
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        # TODO
        symbolicParseDict = parseDict.copy()

        return symbolicParseDict

    @staticmethod
    def computeInputCubeFromOutputCube(outputCube: AbsoluteHyperRectangle, parseDict: Dict) -> HyperRectangle:
        # Get required parameters
        originalInputShape = parseDict['data_in_shape']
        keepDims = parseDict['keepdims']

        # Start from the output cube dimensions and offsets
        in_cube_dims = list(originalInputShape).copy()
        in_cube_offset = [
            0,
        ] * len(in_cube_dims)

        # Iterate through input axes
        out_idx = 0
        for ax in range(len(in_cube_dims)):
            if ax in parseDict['axes']:
                # This axis is reduced
                if keepDims:
                    # Keepdims is set, so the output cube has a dimension here (which will be 1, as it's the reduction result)
                    out_idx += 1
            else:
                # This axis is not reduced, so copy from output cube
                in_cube_dims[ax] = outputCube.dims[out_idx]
                in_cube_offset[ax] = outputCube.offset[out_idx]
                out_idx += 1

        return HyperRectangle(offset = tuple(in_cube_offset), dims = tuple(in_cube_dims))

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # Prepare address names
        addrNames = ['data_in', 'data_out']

        # Extract memory base addresses for each of the required components,
        # based on the computed memory configuration
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        # Prepare replacement lists for the elements inside the operator representation,
        # for the cubes to be computed further down in this function
        replacements: Dict[str, List[int]] = {
            "data_in_shape": [],
            "data_out_shape": [],
            "size": [],
        }

        replacementTypes = {
            "data_in_shape": [
                PointerClass(uint16_t),
                PointerClass(uint16_t),
                PointerClass(uint16_t),
                PointerClass(uint16_t)
            ],
            "data_out_shape": [
                PointerClass(uint16_t),
                PointerClass(uint16_t),
                PointerClass(uint16_t),
                PointerClass(uint16_t)
            ],
            "size": PointerClass(uint16_t),
        }

        # Prepare loading schedule lists
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Iterate over output cubes to compute corresponding input cubes
        for out_cube in [cube.rectangle for cube in absoluteOutputCubes]:
            # Compute input cube
            in_cube = ReduceMeanTileConstraint.computeInputCubeFromOutputCube(out_cube,
                                                                              parseDict = operatorRepresentation)

            # Append replacement elements
            replacements["data_in_shape"].append(list(in_cube.dims).copy())
            replacements["data_out_shape"].append(list(out_cube.dims).copy())
            replacements["size"].append(int(np.prod(out_cube.dims)))

            # Append new cubes
            inputLoadSchedule.append({"data_in": in_cube})
            outputLoadSchedule.append({"data_out": out_cube})

        # Prepare containing objects with information computed in this function regarding tiling schedule
        # and variable replacement inside operator representation
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
