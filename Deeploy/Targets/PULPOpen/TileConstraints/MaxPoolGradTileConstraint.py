# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class MaxPoolGradCTileConstraint(TileConstraint):
    """Channel-tiling constraint for MaxPoolGrad.

    Tiles the channel dimension (last dim in HWC format) across all three tensors:
      - data_in  (grad_output):   [N, Ho, Wo, C]
      - x_in     (original_input):[N, Hi, Wi, C]
      - data_out (grad_input):    [N, Hi, Wi, C]
    All spatial dimensions are kept at their full size.
    """

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        gradOutName = parseDict['data_in']
        xInName = parseDict['x_in']
        gradInName = parseDict['data_out']

        for bufferName in [gradOutName, xInName, gradInName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        numDims = len(ctxt.lookup(gradOutName).shape)

        # All three tensors share the same channel tile size (last dim in HWC)
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = gradInName, dimIdx = numDims -
                                       1) == tilerModel.getTensorDimVar(tensorName = gradOutName, dimIdx = numDims - 1))
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(tensorName = xInName, dimIdx = numDims -
                                       1) == tilerModel.getTensorDimVar(tensorName = gradOutName, dimIdx = numDims - 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        gradOutName = parseDict['data_in']
        xInName = parseDict['x_in']
        gradInName = parseDict['data_out']

        numDims = len(ctxt.lookup(gradOutName).shape)

        # Fix all dimensions except the channel dimension (last) for all three tensors
        for bufferName in [gradOutName, xInName, gradInName]:
            buf_shape = ctxt.lookup(bufferName).shape
            for idx in range(numDims - 1):
                tilerModel.addConstraint(
                    tilerModel.getTensorDimVar(tensorName = bufferName, dimIdx = idx) == buf_shape[idx])

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

        # x_in may or may not be in the tiling solution (if it is a global buffer it is excluded)
        x_in_name = operatorRepresentation['x_in']
        x_in_in_solution = x_in_name in tilingSolution.tensorMemoryConstraints
        if x_in_in_solution:
            xInBaseOffsets, _ = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation, ['x_in'])
            inputBaseOffsets.update(xInBaseOffsets)

        gradOutShape = ctxt.lookup(operatorRepresentation['data_in']).shape
        gradInShape = ctxt.lookup(operatorRepresentation['data_out']).shape
        xInShape = ctxt.lookup(x_in_name).shape

        numDims = len(gradOutShape)

        replacementTypes = {}
        replacements: Dict[str, List[int]] = {}
        replacementTypes["ch_im_in"] = PointerClass(uint16_t)
        replacements["ch_im_in"] = []

        inputInCubes = []
        xInCubes = []

        for cube in outputCubes:
            ch_tile = cube.dims[-1]

            # grad_output tile: same channel slice, full spatial dims
            grad_out_dims = list(gradOutShape)
            grad_out_dims[-1] = ch_tile
            grad_out_offset = list(cube.offset[:-1]) + [cube.offset[-1]]
            inputInCubes.append(HyperRectangle(tuple(grad_out_offset), tuple(grad_out_dims)))

            # x_in tile: same channel slice, full spatial dims
            x_in_dims = list(xInShape)
            x_in_dims[-1] = ch_tile
            x_in_offset = [0] * (numDims - 1) + [cube.offset[-1]]
            xInCubes.append(HyperRectangle(tuple(x_in_offset), tuple(x_in_dims)))

            replacements["ch_im_in"].append(ch_tile)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for grad_out_cube, x_in_cube in zip(inputInCubes, xInCubes):
            entry = {"data_in": grad_out_cube}
            if x_in_in_solution:
                entry["x_in"] = x_in_cube
            inputLoadSchedule.append(entry)

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
