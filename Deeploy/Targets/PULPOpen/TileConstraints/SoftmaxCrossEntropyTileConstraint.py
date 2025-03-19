# ----------------------------------------------------------------------
#
# File: SoftmaxCrossEntropyTileConstraint.py
#
# Last edited: 19.03.2025
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

import numpy as np
from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import HyperRectangle, AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class SoftmaxCrossEntropyGradTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        input1BufferName = parseDict['log_prob']
        input2BufferName = parseDict['labels']
        outputBufferName = parseDict['grad']

        for bufferName in [input1BufferName, input2BufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)
        
        outputDim0 = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        input1Dim0 = tilerModel.getTensorDimVar(tensorName = input1BufferName, dimIdx = 0)
        tilerModel.addConstraint(outputDim0 == input1Dim0)
        outputDim1 = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        input1Dim1 = tilerModel.getTensorDimVar(tensorName = input1BufferName, dimIdx = 1)
        tilerModel.addConstraint(outputDim1 == input1Dim1)
        input2Dim = tilerModel.getTensorDimVar(tensorName = input2BufferName, dimIdx = 0)
        tilerModel.addConstraint(outputDim0 == input2Dim)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['log_prob']
        inputBuffer = ctxt.lookup(inputBufferName)

        lastDimLength = inputBuffer.shape[-1]
        lastDimIdx = len(inputBuffer.shape) - 1
        lastDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx)

        tilerModel.addConstraint(lastDimVar == lastDimLength)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBufferName = parseDict['log_prob']
        inputBuffer = ctxt.lookup(inputBufferName)

        lastDimIdx = len(inputBuffer.shape) - 1

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['num_classes'] = tilerModel.getTensorDimVar(inputBuffer.name, lastDimIdx)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['log_prob', 'labels', 'grad']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"num_classes": [], "batch": []}

        replacementTypes = {"num_classes": PointerClass(uint16_t), "batch": PointerClass(uint16_t)}

        inputlabelCubes = []

        for cube in outputCubes:
            batch = cube.dims[0]
            num_classes = cube.dims[1]

            replacements['num_classes'].append(num_classes)
            replacements['batch'].append(batch)

            labelCube =  HyperRectangle((0, cube.offset[0]), (1, batch))
            inputlabelCubes.append(labelCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for out, label in zip(outputCubes, inputlabelCubes):
            inputLoadSchedule.append({"log_prob": out, "labels": label}) 
            outputLoadSchedule.append({"grad": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
