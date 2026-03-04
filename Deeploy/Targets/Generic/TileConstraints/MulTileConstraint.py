# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

import numpy as np

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, TilingSchedule, VariableReplacementScheme


class MulTileConstraint(TileConstraint):
    """Tile constraint for element-wise Mul.

    Handles the case where B is a scalar/broadcast constant: such a constant is
    excluded from the tiling solution (no L1 allocation needed), so it is
    skipped in the DMA schedule.  When B is a full-sized tensor it is tiled
    together with A and C.
    """

    dataIn1Name = "A"
    dataIn2Name = "B"
    dataOutName = "C"

    @classmethod
    def _B_is_constant(cls, parseDict: Dict, ctxt: NetworkContext) -> bool:
        bName = parseDict[cls.dataIn2Name]
        return isinstance(ctxt.lookup(bName), ConstantBuffer)

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBuffer1Name = parseDict[cls.dataIn1Name]
        outputBufferName = parseDict[cls.dataOutName]

        inputBuffer2Name = parseDict[cls.dataIn2Name]
        bIsConstant = cls._B_is_constant(parseDict, ctxt)

        for bufferName in [inputBuffer1Name, inputBuffer2Name, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        input1Shape = ctxt.lookup(inputBuffer1Name).shape

        for dim in range(len(input1Shape)):
            inputDim1Var = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = dim)
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = dim)
            tilerModel.addConstraint(inputDim1Var == outputDimVar)

            if not bIsConstant:
                inputDim2Var = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name, dimIdx = dim)
                tilerModel.addConstraint(inputDim1Var == inputDim2Var)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        bInSolution = operatorRepresentation[cls.dataIn2Name] in tilingSolution.tensorMemoryConstraints
        addrNames = [cls.dataIn1Name] + ([cls.dataIn2Name] if bInSolution else []) + [cls.dataOutName]
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacements = {"size": []}
        replacementTypes = {"size": PointerClass(uint16_t)}

        for cube in outputCubes:
            replacements["size"].append(np.prod(cube.dims))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            entry = {cls.dataIn1Name: cube}
            if bInSolution:
                entry[cls.dataIn2Name] = cube
            inputLoadSchedule.append(entry)

        for out in outputCubes:
            outputLoadSchedule.append({cls.dataOutName: out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
