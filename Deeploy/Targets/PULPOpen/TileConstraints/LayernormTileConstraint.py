# ----------------------------------------------------------------------
#
# File: LayernormTileConstraint.py
#
# Last edited: 11.09.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author:
# - Run Wang, ETH Zurich
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


class LayernormTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']
        scaleBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        for bufferName in [inputBufferName, outputBufferName, scaleBufferName, biasBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape
        lastDimIdx = len(inputShape) - 1
        lastDimLen = inputShape[-1]

        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == lastDimLen)
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = scaleBufferName, dimIdx = 0))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx) == tilerModel.getTensorDimVar(
                tensorName = biasBufferName, dimIdx = 0))

        for idx, dim in enumerate(inputShape):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = outputBufferName, dimIdx = idx))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]
        addrNames = ['data_in', 'data_out', 'weight', 'bias']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}

        replacementTypes = {"size": PointerClass(uint16_t)}

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)
            weightCube = HyperRectangle((0,), (cube.dims[-1],))
            biasCube = HyperRectangle((0,), (cube.dims[-1],))
            inputLoadSchedule.append({"data_in": cube, "weight": weightCube, "bias": biasCube})
            outputLoadSchedule.append({"data_out": cube})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
