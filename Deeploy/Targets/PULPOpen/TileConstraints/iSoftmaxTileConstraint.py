# ----------------------------------------------------------------------
#
# File: iSoftmaxTileConstraint.py
#
# Last edited: 13.11.2023
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

from typing import Dict, List, Tuple, Union

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class iSoftmaxTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        shapeLen = len(ctxt.lookup(inputBufferName).shape)

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        for idx in range(shapeLen):
            outputDim = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx)
            inputDim = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx)
            tilerModel.addConstraint(outputDim == inputDim)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        inputBuffer = ctxt.lookup(inputBufferName)

        lastDimLength = inputBuffer.shape[-1]
        lastDimIdx = len(inputBuffer.shape) - 1
        lastDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx)

        # tilerModel.addTensorNumOfEltToModel(ctxt, inputBufferName)
        # numVars = tilerModel.getTensorNumberOfEltVar(inputBufferName)

        # tilerModel.addMinTileSizeConstraint(parseDict, 'size', numVars, 8*lastDimLength)

        tilerModel.addConstraint(lastDimVar == lastDimLength)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBufferName = parseDict['data_in']
        inputBuffer = ctxt.lookup(inputBufferName)

        lastDimIdx = len(inputBuffer.shape) - 1

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['lastDimLength'] = tilerModel.getTensorDimVar(inputBuffer.name, lastDimIdx)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"lastDimLength": [], "size": []}

        replacementTypes = {"lastDimLength": PointerClass(uint32_t), "size": PointerClass(uint32_t)}

        for cube in outputCubes:
            lastDimLength = cube.dims[-1]
            size = np.prod(cube.dims)

            replacements['lastDimLength'].append(lastDimLength)
            replacements['size'].append(size)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for out in outputCubes:
            inputLoadSchedule.append({"data_in": out})
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class SoftmaxGradTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        upstream_grad = parseDict['upstream_grad']
        softmax_output = parseDict['softmax_output']
        softmax_grad = parseDict['softmax_grad']

        shapeLen = len(ctxt.lookup(upstream_grad).shape)

        for bufferName in [upstream_grad, softmax_output, softmax_grad]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        for idx in range(shapeLen):
            upstream_dim = tilerModel.getTensorDimVar(tensorName=upstream_grad, dimIdx=idx)
            softmax_out_dim = tilerModel.getTensorDimVar(tensorName=softmax_output, dimIdx=idx)
            softmax_grad_dim = tilerModel.getTensorDimVar(tensorName=softmax_grad, dimIdx=idx)
            
            tilerModel.addConstraint(upstream_dim == softmax_out_dim)
            tilerModel.addConstraint(upstream_dim == softmax_grad_dim)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        upstream_grad = parseDict['upstream_grad']
        inputBuffer = ctxt.lookup(upstream_grad)

        lastDimLength = inputBuffer.shape[-1]
        lastDimIdx = len(inputBuffer.shape) - 1
        lastDimVar = tilerModel.getTensorDimVar(tensorName=upstream_grad, dimIdx=lastDimIdx)

        tilerModel.addConstraint(lastDimVar == lastDimLength)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        upstream_grad = parseDict['upstream_grad']
        inputBuffer = ctxt.lookup(upstream_grad)

        lastDimIdx = len(inputBuffer.shape) - 1

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['lastDimLength'] = tilerModel.getTensorDimVar(upstream_grad, lastDimIdx)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['upstream_grad', 'softmax_output', 'softmax_grad']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                 operatorRepresentation, addrNames)

        replacements = {"lastDimLength": [], "size": []}

        replacementTypes = {"lastDimLength": PointerClass(uint32_t), "size": PointerClass(uint32_t)}

        for cube in outputCubes:
            lastDimLength = cube.dims[-1]
            size = np.prod(cube.dims)

            replacements['lastDimLength'].append(lastDimLength)
            replacements['size'].append(size)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for out in outputCubes:
            inputLoadSchedule.append({"upstream_grad": out, "softmax_output": out})
            outputLoadSchedule.append({"softmax_grad": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule