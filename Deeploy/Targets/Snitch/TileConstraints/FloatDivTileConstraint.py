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
    """Tile constraint for FP32 Div operation with ONNX broadcasting support.

    Supports general NumPy-style broadcasting: both inputs can have any
    dimension, including scalar, partial broadcasting, and full element-wise.
    """

    dataIn1Name = "A"
    dataIn2Name = "B"
    dataOutName = "C"

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer1Name = parseDict[cls.dataIn1Name]
        inputBuffer2Name = parseDict[cls.dataIn2Name]
        outputBufferName = parseDict[cls.dataOutName]

        input1Shape = list(ctxt.lookup(inputBuffer1Name).shape)
        input2Shape = list(ctxt.lookup(inputBuffer2Name).shape)
        outputShape = list(ctxt.lookup(outputBufferName).shape)

        # Add all tensors to model
        tilerModel.addTensorDimToModel(ctxt, inputBuffer1Name)
        tilerModel.addTensorDimToModel(ctxt, inputBuffer2Name)
        tilerModel.addTensorDimToModel(ctxt, outputBufferName)

        outNdim = len(outputShape)

        # Pad input shapes from the left to match output ndim (ONNX broadcasting)
        padded1 = [1] * (outNdim - len(input1Shape)) + input1Shape
        padded2 = [1] * (outNdim - len(input2Shape)) + input2Shape

        for outDim in range(outNdim):
            outputDimVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = outDim)

            # Input 1: map output dim to actual tensor dim
            in1ActualDim = outDim - (outNdim - len(input1Shape))
            if in1ActualDim >= 0:
                in1DimVar = tilerModel.getTensorDimVar(tensorName = inputBuffer1Name, dimIdx = in1ActualDim)
                if padded1[outDim] == 1:
                    tilerModel.addConstraint(in1DimVar == 1)
                else:
                    tilerModel.addConstraint(in1DimVar == outputDimVar)

            # Input 2: map output dim to actual tensor dim
            in2ActualDim = outDim - (outNdim - len(input2Shape))
            if in2ActualDim >= 0:
                in2DimVar = tilerModel.getTensorDimVar(tensorName = inputBuffer2Name, dimIdx = in2ActualDim)
                if padded2[outDim] == 1:
                    tilerModel.addConstraint(in2DimVar == 1)
                else:
                    tilerModel.addConstraint(in2DimVar == outputDimVar)

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

        input1Shape = list(ctxt.lookup(operatorRepresentation[cls.dataIn1Name]).shape)
        input2Shape = list(ctxt.lookup(operatorRepresentation[cls.dataIn2Name]).shape)
        outputShape = list(ctxt.lookup(operatorRepresentation[cls.dataOutName]).shape)

        outNdim = len(outputShape)
        padded1 = [1] * (outNdim - len(input1Shape)) + input1Shape
        padded2 = [1] * (outNdim - len(input2Shape)) + input2Shape

        def _deriveInputCube(outputCube, inputShape, paddedShape):
            """Derive an input HyperRectangle from an output cube, respecting broadcasting."""
            offset = []
            dims = []
            for outDim in range(outNdim):
                actualDim = outDim - (outNdim - len(inputShape))
                if actualDim >= 0:
                    if paddedShape[outDim] == 1:
                        offset.append(0)
                        dims.append(1)
                    else:
                        offset.append(outputCube.offset[outDim])
                        dims.append(outputCube.dims[outDim])
            return HyperRectangle(tuple(offset), tuple(dims))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for cube in outputCubes:
            replacements["size"].append(np.prod(cube.dims))

            in1Cube = _deriveInputCube(cube, input1Shape, padded1)
            in2Cube = _deriveInputCube(cube, input2Shape, padded2)
            inputLoadSchedule.append({cls.dataIn1Name: in1Cube, cls.dataIn2Name: in2Cube})

        for out in outputCubes:
            outputLoadSchedule.append({cls.dataOutName: out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
