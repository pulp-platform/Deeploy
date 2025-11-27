# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class ReduceMeanTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # ===== GET NECESSARY INFORMATION =====
        #   Get I/O buffer names
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        #   Get other necessary information
        reduceAxes = parseDict['axes']
        keepDims = parseDict['keepdims']

        # ===== ADD I/O DIMENSIONS TO THE MODEL AS VARIABLES =====
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # ===== ADD CONSTRAINTS =====
        #   Add constraints for the I/O dimensions
        #   Iterate over input axes and maintain an output index pointer
        inputShape = parseDict['data_in_shape']
        output_idx = 0
        for input_ax in range(len(inputShape)):
            if input_ax in reduceAxes:
                # This axis is reduced
                if keepDims:
                    # Get the output dimension variable and constrain it to 1
                    outputDimensionVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = output_idx)
                    tilerModel.addConstraint(outputDimensionVar == 1)
                    output_idx += 1
                # If keepDims is false, this axis doesn't exist in output, so don't increment output_idx
            else:
                # This axis is not reduced, so input and output dimensions need to be equal
                inputDimensionVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = input_ax)
                outputDimensionVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = output_idx)
                tilerModel.addConstraint(outputDimensionVar == inputDimensionVar)
                output_idx += 1

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:
        symbolicParseDict = parseDict.copy()

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        for ax in range(len(parseDict['data_in_shape'])):
            if ax not in parseDict['axes']:
                dimVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = ax)
                symbolicParseDict['dim_in_' + str(ax)] = dimVar

        return symbolicParseDict

    @staticmethod
    def computeInputCubeFromOutputCube(outputCube: HyperRectangle, parseDict: Dict) -> HyperRectangle:
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

        # Prepare replacements for non-reduced input sizes
        replacements: Dict[str, List[int]] = dict()
        replacementTypes = dict()

        for ax in range(len(operatorRepresentation['data_in_shape'])):
            if ax not in operatorRepresentation['axes']:
                replacements["dim_in_" + str(ax)] = []
                replacementTypes["dim_in_" + str(ax)] = PointerClass(uint32_t)

        # Prepare loading schedule lists
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Iterate over output cubes to compute corresponding input cubes
        for out_cube in [cube.rectangle for cube in absoluteOutputCubes]:
            # Compute input cube
            in_cube = ReduceMeanTileConstraint.computeInputCubeFromOutputCube(out_cube,
                                                                              parseDict = operatorRepresentation)

            # Add replacements for non-reduced input sizes
            for ax in range(len(operatorRepresentation['data_in_shape'])):
                if ax not in operatorRepresentation['axes']:
                    replacements["dim_in_" + str(ax)].append(in_cube.dims[ax])

            # Append new cubes
            inputLoadSchedule.append({"data_in": in_cube})
            outputLoadSchedule.append({"data_out": out_cube})

        # Prepare containing objects with information computed in this function regarding tiling schedule
        # and variable replacement inside operator representation
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
