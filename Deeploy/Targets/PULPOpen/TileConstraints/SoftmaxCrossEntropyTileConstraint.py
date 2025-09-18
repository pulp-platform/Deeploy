# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class SoftmaxCrossEntropyTileConstraint(TileConstraint):

    dataIn1Name = 'logits'
    dataIn2Name = 'labels'
    dataOutName = 'log_prob'

    @classmethod
    def addGeometricalConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        input1BufferName = parseDict[cls.dataIn1Name]
        input2BufferName = parseDict[cls.dataIn2Name]
        outputBufferName = parseDict[cls.dataOutName]

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

    @classmethod
    def addPolicyConstraint(cls, tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict[cls.dataIn1Name]
        inputBuffer = ctxt.lookup(inputBufferName)

        lastDimLength = inputBuffer.shape[-1]
        lastDimIdx = len(inputBuffer.shape) - 1
        lastDimVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = lastDimIdx)

        tilerModel.addConstraint(lastDimVar == lastDimLength)

        return tilerModel

    @classmethod
    def constructSymbolicNodeRep(cls, tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBufferName = parseDict[cls.dataIn1Name]
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

        addrNames = [cls.dataIn1Name, cls.dataIn2Name, cls.dataOutName]
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

            labelCube = HyperRectangle((cube.offset[0],), (batch,))
            inputlabelCubes.append(labelCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for out, label in zip(outputCubes, inputlabelCubes):
            inputLoadSchedule.append({cls.dataIn1Name: out, cls.dataIn2Name: label})
            outputLoadSchedule.append({cls.dataOutName: out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class SoftmaxCrossEntropyGradTileConstraint(SoftmaxCrossEntropyTileConstraint):
    dataIn1Name = 'log_prob'
    dataIn2Name = 'labels'
    dataOutName = 'grad'
