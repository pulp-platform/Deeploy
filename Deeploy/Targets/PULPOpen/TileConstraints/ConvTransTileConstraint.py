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


class ConvTrans2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Geometrical constraint for ConvTranspose, aligned with PULP kernel:
        H_in = (H_out - 1) * stride_h + kernel_h - pad_top - pad_bottom
        W_in = (W_out - 1) * stride_w + kernel_w - pad_left - pad_right
        Layouts:
        input  -> grad_out [N,H_out,W_out,F_total]
        output -> grad_in  [N,H_in,W_in,C]
        weight -> [F_total, P, Q, C]
        """
        inputName  = parseDict['data_in']   # grad_out
        outputName = parseDict['data_out']  # grad_in
        weightName = parseDict['weight']

        tilerModel.addTensorDimToModel(ctxt, inputName)
        tilerModel.addTensorDimToModel(ctxt, outputName)
        tilerModel.addTensorDimToModel(ctxt, weightName)

        pads     = parseDict["pads"]       # [pad_top, pad_bottom, pad_left, pad_right]
        strides  = parseDict["strides"]
        group    = parseDict["group"]

        # NHWC layout
        inH = tilerModel.getTensorDimVar(inputName, 1)
        inW = tilerModel.getTensorDimVar(inputName, 2)
        inC = tilerModel.getTensorDimVar(inputName, 3)

        outH = tilerModel.getTensorDimVar(outputName, 1)
        outW = tilerModel.getTensorDimVar(outputName, 2)
        outC = tilerModel.getTensorDimVar(outputName, 3)

        wOut = tilerModel.getTensorDimVar(weightName, 0)
        wH   = tilerModel.getTensorDimVar(weightName, 1)
        wW   = tilerModel.getTensorDimVar(weightName, 2)
        wIn  = tilerModel.getTensorDimVar(weightName, 3)

        # batch equal
        tilerModel.addConstraint(
            tilerModel.getTensorDimVar(inputName, 0) == tilerModel.getTensorDimVar(outputName, 0)
        )

        # ConvTranspose geometrical relation (C kernel direction)
        expected_outH = (inH - 1) * strides[0] + wH - pads[0] - pads[1]
        expected_outW = (inW - 1) * strides[1] + wW - pads[2] - pads[3]
        tilerModel.addConstraint(outH == expected_outH)
        tilerModel.addConstraint(outW == expected_outW)

        # Channels: input grad_out channels = weight out channels
        tilerModel.addConstraint(inC == wOut)
        tilerModel.addConstraint(outC == wIn * group)

        return tilerModel


    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        """
        Add policy constraints for ConvTranspose tiling.
        
        Key constraints:
        - Input channels must be complete (can't tile input channels)
        - Kernel dimensions must be complete
        - Output spatial dims can be tiled (but input must accommodate)
        """

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        pads = parseDict["pads"]
        strides = parseDict["strides"]

        # Extract dimension variables
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)

        # Input channels must be complete (no tiling on input channels)
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])

        # Kernel must not be tiled
        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])
        tilerModel.addConstraint(weightInChannelVar * parseDict['group'] == parseDict['ch_im_out'])

        # Minimum input tile size: at least 1 pixel
        # (ConvTranspose can work with single input pixel)
        tilerModel.addConstraint(inputHeightVar >= 1)
        tilerModel.addConstraint(inputWidthVar >= 1)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        symbolicParseDict = parseDict.copy()

        symbolicParseDict['dim_im_in_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_im_in_y'] = tilerModel.getTensorDimVar(inputBuffer.name, 2)

        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)

        symbolicParseDict['dim_im_out_x'] = tilerModel.getTensorDimVar(outputBuffer.name, 1)
        symbolicParseDict['dim_im_out_y'] = tilerModel.getTensorDimVar(outputBuffer.name, 2)

        return symbolicParseDict

    @staticmethod
    def computeInputCube(
        kernelShape: Tuple[int, int],
        pads: Tuple[int, int, int, int],   # (pad_top, pad_bottom, pad_left, pad_right)
        strides: Tuple[int, int],
        inputCSize: int,
        outputCube: HyperRectangle,
        outputDims: Tuple[int, int, int, int],
        inputDims: Tuple[int, int, int, int],
        ) -> Tuple[HyperRectangle, Tuple[int, int, int, int]]:
        """
        Compute corresponding input (grad_out) tile from an output (grad_in) tile.
        Aligned with:
            H_in = (H_out - 1)*SP + P - pad_top - pad_bottom
        """

        (outBatchOff, outHOff, outWOff, _) = outputCube.offset
        (outBatchSize, outHSize, outWSize, _) = outputCube.dims

        pad_top, pad_bottom, pad_left, pad_right = pads
        stride_h, stride_w = strides
        kernel_h, kernel_w = kernelShape

        # OutputCube covers [outHOff : outHOff+outHSize)
        # Compute minimal input coverage to produce it
        inHOff = outHOff // stride_h
        inWOff = outWOff // stride_w

        inHEnd = (outHOff + outHSize - 1) // stride_h + 1
        inWEnd = (outWOff + outWSize - 1) // stride_w + 1

        inHSize = inHEnd - inHOff
        inWSize = inWEnd - inWOff

        inHSize = min(inHSize, inputDims[1] - inHOff)
        inWSize = min(inWSize, inputDims[2] - inWOff)
        inHSize = max(1, inHSize)
        inWSize = max(1, inWSize)

        # Compute tile-level padding
        tile_pad_top    = pad_top    if outHOff == 0 else 0
        tile_pad_bottom = pad_bottom if (outHOff + outHSize) == outputDims[1] else 0
        tile_pad_left   = pad_left   if outWOff == 0 else 0
        tile_pad_right  = pad_right  if (outWOff + outWSize) == outputDims[2] else 0

        InCube = HyperRectangle(
            (outBatchOff, inHOff, inWOff, 0),
            (outBatchSize, inHSize, inWSize, inputCSize)
        )

        # Return pads in (pad_top, pad_bottom, pad_left, pad_right)
        return InCube, (tile_pad_top, tile_pad_bottom, tile_pad_left, tile_pad_right)

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        varIn = operatorRepresentation["data_in"]
        varWeight = operatorRepresentation['weight']
        varOut = operatorRepresentation['data_out']

        group = operatorRepresentation["group"]

        addrNames = ['data_in', 'weight', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inputInCubes = []
        inputWeightCubes = []

        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_in": [],
            "ch_im_out": [],
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": []
        }

        replacementTypes = {
            "dim_im_in_x": PointerClass(uint16_t),
            "dim_im_in_y": PointerClass(uint16_t),
            "dim_im_out_x": PointerClass(uint16_t),
            "dim_im_out_y": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "ch_im_out": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }


        (weightOutCh, weightH, weightW, weightInCh) = ctxt.lookup(varWeight).shape

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        for cube in outputCubes:
            COffset = cube.offset[3]
            (_, HSize, WSize, CSize) = cube.dims

            # Compute input cube (smaller) for this output cube (larger)
            InCube, padding_tuple = ConvTrans2DTileConstraint.computeInputCube(
                kernelShape = (weightH, weightW),
                pads = pads,
                strides = strides,
                inputCSize = weightOutCh,  # ConvTranspose: input uses weight output channels
                outputCube = cube,
                inputDims = ctxt.lookup(varIn).shape,
                outputDims = ctxt.lookup(varOut).shape)

            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_in_y'].append(InCube.dims[2])

            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)

            replacements['ch_im_in'].append(weightOutCh)
            replacements['ch_im_out'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(InCube)

            WeightCube = HyperRectangle((0, 0, 0, COffset), (weightOutCh, weightH, weightW, CSize))
            inputWeightCubes.append(WeightCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b in zip(inputInCubes, inputWeightCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
