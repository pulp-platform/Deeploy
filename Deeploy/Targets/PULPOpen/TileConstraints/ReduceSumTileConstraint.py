# ----------------------------------------------------------------------
#
# File: ReduceSumTileConstraint.py
#
# Last edited: 09.06.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
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

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class ReduceSumTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        inputBuffer = ctxt.lookup(inputBufferName)
        outputBuffer = ctxt.lookup(outputBufferName)

        inputShapeLen = len(inputBuffer.shape)
        outputShapeLen = len(outputBuffer.shape)

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # For ReduceSum, we need to handle dimension reduction
        # If keepdims=True, all dimensions should match (reduced dims become 1)
        # If keepdims=False, reduced dimensions are removed from output

        keepdims = parseDict.get('keepdims', True)  # Default to True if not specified

        if keepdims:
            # keepdims=True: output has same number of dimensions as input
            if inputShapeLen == outputShapeLen:
                for idx in range(inputShapeLen):
                    outputDim = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)
                    inputDim = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx)

                    # For reduced dimensions, output should be 1
                    if 'axis' in parseDict:
                        axis = parseDict['axis']
                        if isinstance(axis, int):
                            axis = [axis]

                        # Handle negative axis indexing
                        normalized_axis = []
                        for ax in axis:
                            if ax < 0:
                                ax = inputShapeLen + ax
                            normalized_axis.append(ax)

                        if idx in normalized_axis:
                            # This dimension is reduced, output should be 1
                            tilerModel.addConstraint(outputDim == 1)
                        else:
                            # This dimension is preserved
                            tilerModel.addConstraint(outputDim == inputDim)
                    else:
                        # No axis specified, all dimensions are reduced to 1
                        tilerModel.addConstraint(outputDim == 1)
            else:
                raise ValueError("With keepdims=True, input and output should have same number of dimensions")

        else:
            # keepdims=False: reduced dimensions are removed from output
            if 'axis' in parseDict:
                axis = parseDict['axis']
                if isinstance(axis, int):
                    axis = [axis]

                # Handle negative axis indexing
                normalized_axis = []
                for ax in axis:
                    if ax < 0:
                        ax = inputShapeLen + ax
                    normalized_axis.append(ax)
                normalized_axis = sorted(normalized_axis)

                # Expected output shape length
                expected_output_len = inputShapeLen - len(normalized_axis)

                if outputShapeLen != expected_output_len:
                    raise ValueError(f"With keepdims=False and axis={axis}, expected output to have "
                                     f"{expected_output_len} dimensions, but got {outputShapeLen}")

                # Map input dimensions to output dimensions (skipping reduced ones)
                output_idx = 0
                for input_idx in range(inputShapeLen):
                    if input_idx not in normalized_axis:
                        # This dimension is preserved
                        outputDim = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = output_idx)
                        inputDim = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = input_idx)
                        tilerModel.addConstraint(outputDim == inputDim)
                        output_idx += 1

            else:
                # No axis specified - global reduction, output should be scalar
                # In many frameworks, scalar outputs are represented as 1D tensors with size 1
                # or as 0D tensors (empty shape)
                if outputShapeLen == 0:
                    # True scalar (0D tensor) - nothing to constrain
                    pass
                elif outputShapeLen == 1:
                    # 1D tensor with size 1 representing scalar
                    outputDim = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
                    tilerModel.addConstraint(outputDim == 1)
                else:
                    # Allow other representations but warn about potential issues
                    # Some frameworks might represent scalars differently
                    # For now, just ensure all output dimensions are 1
                    for idx in range(outputShapeLen):
                        outputDim = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)
                        tilerModel.addConstraint(outputDim == 1)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # No constraints - let the tiler handle dimensions normally
        # We'll handle the actual ReduceSum logic in serializeTilingSolution
        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBufferName = parseDict['data_in']
        inputBuffer = ctxt.lookup(inputBufferName)

        symbolicParseDict = parseDict.copy()

        # Since we force all dimensions to be full size, we can use the actual shape
        # This ensures the template gets the correct dimensions for the single cube
        symbolicParseDict['data_in_shape'] = list(inputBuffer.shape)

        # Add axes information (normalized)
        if 'axis' in parseDict:
            axis = parseDict['axis']
            if isinstance(axis, int):
                axes = [axis]
            else:
                axes = list(axis)

            # Handle negative axis indexing
            normalized_axes = []
            for ax in axes:
                if ax < 0:
                    ax = len(inputBuffer.shape) + ax
                normalized_axes.append(ax)

            symbolicParseDict['axes'] = normalized_axes
        else:
            # Global reduction - all axes
            symbolicParseDict['axes'] = list(range(len(inputBuffer.shape)))

        # Add keepdims information
        symbolicParseDict['keepdims'] = parseDict.get('keepdims', True)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        # Get original tensor shapes from context
        inputBufferName = operatorRepresentation['data_in']
        outputBufferName = operatorRepresentation['data_out']
        inputBuffer = ctxt.lookup(inputBufferName)
        outputBuffer = ctxt.lookup(outputBufferName)

        # Use original dimensions for ReduceSum computation
        originalInputShape = list(inputBuffer.shape)
        originalOutputShape = list(outputBuffer.shape)

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"data_in_shape": [], "axes": [], "keepdims": [], "reduceLength": []}
        replacementTypes = {
            "data_in_shape": PointerClass(uint32_t),
            "axes": PointerClass(uint32_t),
            "keepdims": PointerClass(uint32_t),
            "reduceLength": PointerClass(uint32_t)
        }

        # Get axis and keepdims information from operator representation
        # Note: the key might be 'axes' (plural) instead of 'axis' (singular)
        axis = operatorRepresentation.get('axis', operatorRepresentation.get('axes', None))
        keepdims = operatorRepresentation.get('keepdims', True)

        # Calculate axes (normalize negative indices)
        if axis is not None:
            if isinstance(axis, int):
                axes = [axis]
            else:
                axes = list(axis)

            # Handle negative axis indexing
            normalized_axes = []
            for ax in axes:
                if ax < 0:
                    ax = len(originalInputShape) + ax
                normalized_axes.append(ax)
            axes = normalized_axes
        else:
            # Global reduction - all axes
            axes = list(range(len(originalInputShape)))

        # Calculate reduceLength (product of dimensions being reduced)
        reduceLength = 1
        for ax in axes:
            reduceLength *= originalInputShape[ax]

        # For ReduceSum, we always use the original tensor dimensions
        # regardless of how the tiler decides to split them
        replacements['data_in_shape'].append(tuple(originalInputShape))
        replacements['axes'].append(tuple(axes))
        replacements['keepdims'].append(1 if keepdims else 0)
        replacements['reduceLength'].append(reduceLength)

        # Create scheduling based on original dimensions
        inputLoadSchedule = []
        outputLoadSchedule = []

        # Create HyperRectangles with original dimensions
        from Deeploy.TilingExtension.TilingCodegen import HyperRectangle

        inputCube = HyperRectangle(dims = originalInputShape, offset = [0] * len(originalInputShape))

        outputCube = HyperRectangle(dims = originalOutputShape, offset = [0] * len(originalOutputShape))

        inputLoadSchedule.append({"data_in": inputCube})
        outputLoadSchedule.append({"data_out": outputCube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
