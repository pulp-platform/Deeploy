# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    _invertPermutation, _permuteHyperRectangle
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class TransposeTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # Map output dims to inputs dims
        for idx, perm_idx in enumerate(parseDict["perm"]):
            tilerModel.addConstraint(
                tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = idx) == tilerModel.getTensorDimVar(
                    tensorName = inputBufferName, dimIdx = perm_idx))

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        replacementTypes = {}
        replacements: Dict[str, List[int]] = {}

        numDims = len(ctxt.lookup(operatorRepresentation['data_in']).shape)

        for dim in range(numDims):
            replacementTypes[f"dimLen_{dim}"] = PointerClass(uint16_t)
            replacements[f"dimLen_{dim}"] = []

        invPerm = _invertPermutation(operatorRepresentation['perm'])
        inputCubes = []
        for outCube in outputCubes:
            # Workaround: MatMulLayer.computeShapes may inject leading batch dims into
            # the Transpose output's ctxt shape (e.g. [K,N] → [1,K,N]) when A is 3-D.
            # The actual Transpose only operates on the spatial (last len(invPerm)) dims;
            # strip the extra leading batch dims before permuting so that inCube matches
            # the real data_in (weight) shape which is not broadened.
            numExtra = len(outCube.dims) - len(invPerm)
            if numExtra > 0:
                spatialCube = HyperRectangle(outCube.offset[numExtra:], outCube.dims[numExtra:])
                inCube = _permuteHyperRectangle(spatialCube, invPerm)
            else:
                inCube = _permuteHyperRectangle(outCube, invPerm)
            inputCubes.append(inCube)
            for i, dim in enumerate(inCube.dims):
                replacements[f"dimLen_{i}"].append(dim)

        inputLoadSchedule = [{"data_in": cube} for cube in inputCubes]
        outputLoadSchedule = [{"data_out": cube} for cube in outputCubes]
        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
