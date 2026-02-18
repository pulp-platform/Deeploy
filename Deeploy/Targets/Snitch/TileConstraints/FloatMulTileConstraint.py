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


class FloatMulTileConstraint(TileConstraint):
    """Tile constraint for FP32 Mul: supports scalar and element-wise cases."""

    dataIn1Name = "A"
    dataIn2Name = "B"
    dataOutName = "C"

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer1Name = parseDict[cls.dataIn1Name]
        inputBuffer2Name = parseDict[cls.dataIn2Name]
        outputBufferName = parseDict[cls.dataOutName]

        tilerModel.addTensorDimToModel(ctxt, inputBuffer1Name)
        tilerModel.addTensorDimToModel(ctxt, inputBuffer2Name)
        tilerModel.addTensorDimToModel(ctxt, outputBufferName)

        input1Shape = list(ctxt.lookup(inputBuffer1Name).shape)
        input2Shape = list(ctxt.lookup(inputBuffer2Name).shape)

        is_scalar = (np.prod(input2Shape) == 1)

        if is_scalar:
            # Scalar: tile A and C together, B stays fixed
            for dim in range(len(input1Shape)):
                in1Var = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = dim)
                outVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
                tilerModel.addConstraint(in1Var == outVar)
            for dim in range(len(input2Shape)):
                in2Var = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name, dimIdx = dim)
                tilerModel.addConstraint(in2Var == input2Shape[dim])
        else:
            # Element-wise: all three tensors tiled identically
            for dim in range(len(input1Shape)):
                in1Var = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = dim)
                in2Var = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name, dimIdx = dim)
                outVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
                tilerModel.addConstraint(in1Var == in2Var)
                tilerModel.addConstraint(in1Var == outVar)

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

        input2Shape = list(ctxt.lookup(operatorRepresentation[cls.dataIn2Name]).shape)
        is_scalar = (np.prod(input2Shape) == 1)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            replacements["size"].append(np.prod(cube.dims))
            if is_scalar:
                in2Cube = HyperRectangle(tuple([0] * len(input2Shape)), tuple(input2Shape))
                inputLoadSchedule.append({cls.dataIn1Name: cube, cls.dataIn2Name: in2Cube})
            else:
                inputLoadSchedule.append({cls.dataIn1Name: cube, cls.dataIn2Name: cube})

        for out in outputCubes:
            outputLoadSchedule.append({cls.dataOutName: out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
