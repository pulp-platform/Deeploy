# ----------------------------------------------------------------------
#
# File: ConcatTileConstraint.py
#
# Last edited: 19.02.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
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

import copy
from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class ConcatTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer1Name = parseDict['data_in_1']
        inputBuffer2Name = parseDict['data_in_2']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBuffer1Name, inputBuffer2Name, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        input1Shape = ctxt.lookup(inputBuffer1Name).shape
        outputShape = ctxt.lookup(outputBufferName).shape

        axis = parseDict['axis']
        posAxis = axis if axis >= 0 else len(input1Shape) + axis

        for dim in range(len(input1Shape)):
            inputDim1Var = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = dim)
            inputDim2Var = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)

            if dim == posAxis:
                tilerModel.addConstraint(inputDim1Var + inputDim2Var == outputDimVar)
            else:
                tilerModel.addConstraint(inputDim1Var == outputDimVar)
                tilerModel.addConstraint(inputDim2Var == inputDim1Var)

        for dim in range(posAxis, len(input1Shape), 1):

            inputDim1Var = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = dim)
            inputDim2Var = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)

            tilerModel.addConstraint(inputDim1Var == input1Shape[dim])
            tilerModel.addConstraint(outputDimVar == outputShape[dim])

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in_1', 'data_in_2', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"iterations": [], "in1TransferLength": [], "in2TransferLength": []}
        replacementTypes = {
            "iterations": PointerClass(uint16_t),
            "in1TransferLength": PointerClass(uint16_t),
            "in2TransferLength": PointerClass(uint16_t)
        }

        in1Shape = ctxt.lookup(operatorRepresentation['data_in_1']).shape
        in2Shape = ctxt.lookup(operatorRepresentation['data_in_2']).shape

        dataIn1 = ctxt.lookup(operatorRepresentation['data_in_1'])
        dataIn2 = ctxt.lookup(operatorRepresentation['data_in_2'])

        axis = operatorRepresentation['axis']
        posAxis = axis if axis >= 0 else len(in1Shape) + axis

        for cube in outputCubes:
            newIterations = np.prod(cube.dims[:axis])
            replacements["iterations"].append(newIterations)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:

            in1Cube = copy.deepcopy(cube)
            in2Cube = copy.deepcopy(cube)

            if posAxis < (len(in1Shape) - 1):
                in1Cube.dims = (*in1Cube.dims[:posAxis], in1Shape[posAxis], *in1Cube.dims[posAxis + 1:])
                in2Cube.dims = (*in2Cube.dims[:posAxis], in2Shape[posAxis], *in2Cube.dims[posAxis + 1:])

            else:
                in1Cube.dims = (*in1Cube.dims[:posAxis], in1Shape[posAxis])
                in2Cube.dims = (*in2Cube.dims[:posAxis], in2Shape[posAxis])

            replacements["in1TransferLength"].append(
                np.prod(in1Cube.dims[posAxis:]) * (dataIn1._type.referencedType.typeWidth // 8))
            replacements["in2TransferLength"].append(
                np.prod(in2Cube.dims[posAxis:]) * (dataIn2._type.referencedType.typeWidth // 8))

            inputLoadSchedule.append({"data_in_1": in1Cube, "data_in_2": in2Cube})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
