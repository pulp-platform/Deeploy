# ----------------------------------------------------------------------
#
# File: iNoNormTileConstraint.py
#
# Last edited: 06.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme

class iNoNormTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        weightsBufferName = parseDict['weights']
        biasBufferName = parseDict['bias']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, weightsBufferName, biasBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputShape = ctxt.lookup(inputBufferName).shape

        weigthsBufferShapeLen = len(ctxt.lookup(weightsBufferName).shape)
        biasBufferShapeLen = len(ctxt.lookup(biasBufferName).shape)

        weightsLastDimVar = tilerModel.getTensorDimVar(tensorName = weightsBufferName,
                                                       dimIdx = weigthsBufferShapeLen - 1)
        biasLastDimVar = tilerModel.getTensorDimVar(tensorName = biasBufferName, dimIdx = biasBufferShapeLen - 1)

        tilerModel.addConstraint(biasLastDimVar == weightsLastDimVar)

        for dim in range(len(inputShape)):
            inputDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = dim)
            weightDimVar = tilerModel.getTensorDimVar(tensorName = weightsBufferName, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
            tilerModel.addConstraint(inputDimVar == outputDimVar)
            tilerModel.addConstraint(weightDimVar == outputDimVar)

        return tilerModel

    @classmethod
    def serializeTilingSolution(cls, tilingSolution: NodeMemoryConstraint,
                                absoluteOutputCubes: List[AbsoluteHyperRectangle], targetMemLevel: str,
                                ctxt: NetworkContext,
                                operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weights', 'bias', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation, addrNames)

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint32_t)}

        inputCubes = []
        weightCubes = []
        biasCubes = []

        for outputCube in outputCubes:

            size = np.prod(outputCube.dims[1:])
            lastDimLength = outputCube.dims[-1]

            replacements['size'].append(size)

            inputCubes.append(outputCube)
            weightCubes.append(outputCube)
            biasCubes.append(outputCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for inp, w, b in zip(inputCubes, weightCubes, biasCubes):
            inputLoadSchedule.append({"data_in": inp, "weights": w, "bias": b})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
