# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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


class FloatDivTileConstraint(TileConstraint):
    """Tile constraint for FP32 Div operation supporting scalar broadcasting."""

    dataIn1Name = "input1"
    dataIn2Name = "input2"
    dataOutName = "output"

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer1Name = parseDict[cls.dataIn1Name]
        inputBuffer2Name = parseDict[cls.dataIn2Name]
        outputBufferName = parseDict[cls.dataOutName]

        input1Shape = ctxt.lookup(inputBuffer1Name).shape
        input2Shape = ctxt.lookup(inputBuffer2Name).shape

        # Add tensor dimensions to model
        tilerModel.addTensorDimToModel(ctxt, inputBuffer1Name)
        tilerModel.addTensorDimToModel(ctxt, outputBufferName)

        # Check if input2 is scalar (total size == 1)
        is_scalar = np.prod(input2Shape) == 1

        if is_scalar:
            # Scalar broadcasting: input2 is a scalar, don't tile it
            # Only add input2 dimensions if it has more than 0 dims
            if len(input2Shape) > 0:
                tilerModel.addTensorDimToModel(ctxt, inputBuffer2Name)
                # Constrain scalar to remain untiled (size 1)
                for dim in range(len(input2Shape)):
                    input2DimVar = tilerModel.getTensorDimVar(tensorName=inputBuffer2Name, dimIdx=dim)
                    tilerModel.addConstraint(input2DimVar == input2Shape[dim])

            # Input1 and output must have same dimensions
            for dim in range(len(input1Shape)):
                inputDim1Var = tilerModel.getTensorDimVar(tensorName=inputBuffer1Name, dimIdx=dim)
                outputDimVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=dim)
                tilerModel.addConstraint(inputDim1Var == outputDimVar)
        else:
            # Element-wise: both inputs must have same shape
            tilerModel.addTensorDimToModel(ctxt, inputBuffer2Name)

            for dim in range(len(input1Shape)):
                inputDim1Var = tilerModel.getTensorDimVar(tensorName=inputBuffer1Name, dimIdx=dim)
                inputDim2Var = tilerModel.getTensorDimVar(tensorName=inputBuffer2Name, dimIdx=dim)
                outputDimVar = tilerModel.getTensorDimVar(tensorName=outputBufferName, dimIdx=dim)

                tilerModel.addConstraint(inputDim1Var == inputDim2Var)
                tilerModel.addConstraint(inputDim1Var == outputDimVar)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = [cls.dataIn1Name, cls.dataIn2Name, cls.dataOutName]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint16_t)}

        # Check if scalar broadcasting
        input2Name = operatorRepresentation[cls.dataIn2Name]
        input2Shape = ctxt.lookup(input2Name).shape
        is_scalar = np.prod(input2Shape) == 1

        for cube in outputCubes:
            newSize = np.prod(cube.dims)
            replacements["size"].append(newSize)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            if is_scalar:
                # For scalar, load the entire scalar tensor (size 1)
                scalarCube = HyperRectangle(tuple([0] * len(input2Shape)), tuple(input2Shape))
                inputLoadSchedule.append({cls.dataIn1Name: cube, cls.dataIn2Name: scalarCube})
            else:
                inputLoadSchedule.append({cls.dataIn1Name: cube, cls.dataIn2Name: cube})

        for out in outputCubes:
            outputLoadSchedule.append({cls.dataOutName: out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
