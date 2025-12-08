# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class ConvGradW2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Geometrical constraint for ConvGradW (Weight Gradient).
        Computes gradient of weights from output gradient and input activations.

        Layouts:
        data_in (grad_out) -> [N, H_out, W_out, C_out]
        weight (input_act) -> [N, H_in, W_in, C_in]
        data_out (grad_w)  -> [C_out, K_h, K_w, C_in]
        """
        inputName  = parseDict['data_in']   # grad_out
        outputName = parseDict['data_out']  # grad_weight
        weightName = parseDict['weight']    # input activations

        tilerModel.addTensorDimToModel(ctxt, inputName)
        tilerModel.addTensorDimToModel(ctxt, outputName)
        tilerModel.addTensorDimToModel(ctxt, weightName)

        pads     = parseDict["pads"]
        strides  = parseDict["strides"]
        group    = parseDict["group"]

        # NHWC layout
        # input (grad_out): [N, H_out, W_out, C_out]
        inH = tilerModel.getTensorDimVar(inputName, 1)
        inW = tilerModel.getTensorDimVar(inputName, 2)
        inC = tilerModel.getTensorDimVar(inputName, 3)

        # weight (input activations): [N, H_in, W_in, C_in]
        wH = tilerModel.getTensorDimVar(weightName, 1)
        wW = tilerModel.getTensorDimVar(weightName, 2)
        wC = tilerModel.getTensorDimVar(weightName, 3)

        # output (grad_weight): [C_out, K_h, K_w, C_in]
        outC = tilerModel.getTensorDimVar(outputName, 0)
        outH = tilerModel.getTensorDimVar(outputName, 1)
        outW = tilerModel.getTensorDimVar(outputName, 2)
        outCh = tilerModel.getTensorDimVar(outputName, 3)

        # batch equal for input tensors
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(inputName, 0) == tilerModel.getTensorDimVar(weightName, 0)
        )

        # Kernel dimensions
        kernel_h = parseDict['dim_kernel_x']
        kernel_w = parseDict['dim_kernel_y']

        tilerModel.addConstraint(outH == kernel_h)
        tilerModel.addConstraint(outW == kernel_w)

        # Channels
        tilerModel.addConstraint(inC == outC)
        tilerModel.addConstraint(wC == outCh * group)

        # Forward conv relation: H_out = (H_in + pad - K) / stride + 1
        expected_outH = (wH + pads[0] + pads[1] - kernel_h) // strides[0] + 1
        expected_outW = (wW + pads[2] + pads[3] - kernel_w) // strides[1] + 1
        tilerModel.addConstraint(inH == expected_outH)
        tilerModel.addConstraint(inW == expected_outW)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Add policy constraints for ConvGradW tiling.

        Key constraints:
        - Kernel dimensions and output channels must be complete
        - Input channels must be complete
        - Spatial dimensions can be tiled
        """

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        # Output channels must be complete (no tiling on output channels)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 0)
        tilerModel.addConstraint(outputChannelVar == parseDict['ch_im_out'])

        # Kernel dimensions must not be tiled
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)
        tilerModel.addConstraint(outputHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(outputWidthVar == parseDict['dim_kernel_y'])

        # Output input channels must be complete
        outputInChannelVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 3)
        tilerModel.addConstraint(outputInChannelVar * parseDict['group'] == parseDict['ch_im_in'])

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        symbolicParseDict = parseDict.copy()

        # grad_out dimensions
        symbolicParseDict['dim_im_out_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_im_out_y'] = tilerModel.getTensorDimVar(inputBuffer.name, 2)

        # input activation dimensions
        symbolicParseDict['dim_im_in_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_im_in_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)

        # kernel dimensions (from output)
        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(outputBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(outputBuffer.name, 2)

        return symbolicParseDict

    @staticmethod
    def serializeTilingSolution(tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
                                 targetMemLevel: str, ctxt: NetworkContext,
                                 operatorRepresentation: OperatorRepresentation) -> TilingSchedule:

        # For simplicity, use basic serialization
        # In production, you might need custom logic
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'data_out']
        inputBaseOffsets, outputBaseOffsets = TileConstraint.extractBaseOffsets(tilingSolution, targetMemLevel,
                                                                                  addrNames)

        varWeight = operatorRepresentation['weight']
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        inputWeightCubes = []

        for cube in outputCubes:
            # For now, use full input cubes
            # In production, compute proper input tiles based on the computation
            inputInCubes.append(HyperRectangle((0, 0, 0, 0),
                                               ctxt.lookup(operatorRepresentation['data_in']).shape))
            inputWeightCubes.append(HyperRectangle((0, 0, 0, 0),
                                                    ctxt.lookup(operatorRepresentation['weight']).shape))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b, c in zip(inputInCubes, inputWeightCubes, outputCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b})
            outputLoadSchedule.append({"data_out": c})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule,
                                        tilingSolution)

        return tilingSchedule
