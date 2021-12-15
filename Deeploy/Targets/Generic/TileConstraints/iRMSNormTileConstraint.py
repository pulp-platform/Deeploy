# ----------------------------------------------------------------------
#
# File: iRMSNormTileConstraint.py
#
# Last edited: 21.02.2024
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


class iRMSNormTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputName = parseDict['data_in']
        weightName = parseDict['weight']
        outputName = parseDict['data_out']

        for bufferName in [inputName, weightName, outputName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputName).shape
        lastDimIdx = len(inputShape) - 1
        lastDimLen = inputShape[-1]

        tilerModel.addConstraint(tilerModel.getTensorDimVar(tensorName = inputName, dimIdx = lastDimIdx) == lastDimLen)
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = weightName, dimIdx = 0))

        for idx, dim in enumerate(inputShape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = inputName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = outputName, dimIdx = idx))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint16_t)}

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:

            weightCube = copy.deepcopy(cube)
            weightCube.dims = (cube.dims[-1],)
            inputLoadSchedule.append({"data_in": cube, "weight": weightCube})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
