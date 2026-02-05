# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
            outputDim = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=idx)
            inputDim = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=idx)
            tilerModel.addConstraint(outputDim == inputDim)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        inputBuffer = ctxt.lookup(inputBufferName)

        lastDimLength = inputBuffer.shape[-1]
        lastDimIdx = len(inputBuffer.shape) - 1
        lastDimVar = tilerModel.getTensorDimVar(tensorName=inputBufferName, dimIdx=lastDimIdx)

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
