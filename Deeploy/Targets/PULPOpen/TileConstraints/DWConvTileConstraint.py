# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class RQDWConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        '''
        This function add geometrical constraints for a PULP Im2Col Convolution Tilling.
        '''

        # Get to-be-tiled tensor's buffers
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']
        outputBufferName = parseDict['data_out']

        strides = parseDict["strides"]
        padding = parseDict["pads"]
        dilation = parseDict["dilations"]

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, weightBufferName, mulBufferName, addBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        # SCHEREMO: NCHW layout

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)

        # SCHEREMO: CHW layout

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)

        # SCHEREMO: NHWC layout

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        # map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)  # Output Channel
        tilerModel.addConstraint(outputChannelVar == inputChannelVar)  # Input Channel

        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        inputBuffer = ctxt.lookup(inputBufferName)

        effectiveHeight = inputHeightVar + ((padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[2]))
        effectiveWidth = inputWidthVar + ((padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[3]))

        tilerModel.addConstraint((outputHeightVar == (effectiveHeight - (weightHeightVar - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint((outputWidthVar == (effectiveWidth - (weightWidthVar - 1) - 1) // strides[1] + 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)

        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)

        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)

        # SCHEREMO: Work around tiling issue with non-wordaligned accesses
        if "L3" in ctxt.lookup(parseDict['data_in'])._memoryLevel:
            tilerModel.addTileSizeDivisibleConstraint(parseDict, 'ch_im_in', inputChannelVar, 4)

        strides = parseDict["strides"]
        pads = parseDict["pads"]

        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])

        # VIC: Constraint the minimum tile size such that we can apply at least one kernel on it
        # SCHEREMO: Account for padding - needed for MobileNetv1
        # JS: Clamp the minimum to the full (untiled) spatial dimension: when the kernel is
        # larger than the feature map (e.g. a 7x7 DWConv on a 3x3 late-stage map), the
        # "one full kernel" minimum would exceed the available size and make the model
        # infeasible. In that case the only valid tile is the whole dimension.
        tilerModel.addConstraint(outputHeightVar >= min(1 + max([pads[0], pads[2]]), outputBuffer.shape[1]))
        tilerModel.addConstraint(outputWidthVar >= min(1 + max([pads[1], pads[3]]), outputBuffer.shape[2]))

        tilerModel.addConstraint(inputHeightVar >= min(parseDict['dim_kernel_x'] + pads[0], inputBuffer.shape[2]),
                                 strategy = PerformanceHint(1))
        tilerModel.addConstraint(inputWidthVar >= min(parseDict['dim_kernel_y'] + pads[1], inputBuffer.shape[3]),
                                 strategy = PerformanceHint(1))

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['ch_im_in'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)

        return symbolicParseDict

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['data_in', 'weight', 'mul', 'add', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varWeight = operatorRepresentation['weight']
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        inputAddCubes = []
        inputMulCubes = []
        inputWeightCubes = []
        replacements: Dict[str, List[int]] = {
            "dim_im_in_x": [],
            "dim_im_in_y": [],
            "dim_im_out_x": [],
            "dim_im_out_y": [],
            "ch_im_out": [],
            "ch_im_in": [],
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
            "ch_im_out": PointerClass(uint16_t),
            "ch_im_in": PointerClass(uint16_t),
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }

        weightH = ctxt.lookup(varWeight).shape[1]
        weightW = ctxt.lookup(varWeight).shape[2]

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        for cube in outputCubes:

            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            NHWCInCube, padding_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, CSize,
                                                                              cube,
                                                                              ctxt.lookup(varOut).shape)
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            NCHWInCube = HyperRectangle((NHWCInCube.offset[0], COffset, NHWCInCube.offset[1], NHWCInCube.offset[2]),
                                        (NHWCInCube.dims[0], CSize, NHWCInCube.dims[1], NHWCInCube.dims[2]))

            RequantCube = HyperRectangle((COffset,), (CSize,))
            WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, weightH, weightW, 1))

            replacements['dim_im_in_x'].append(NCHWInCube.dims[2])
            replacements['dim_im_in_y'].append(NCHWInCube.dims[3])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)
            replacements['ch_im_in'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(NCHWInCube)
            inputAddCubes.append(RequantCube)
            inputMulCubes.append(RequantCube)
            inputWeightCubes.append(WeightCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b, add, mul in zip(inputInCubes, inputWeightCubes, inputAddCubes, inputMulCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b, "add": add, "mul": mul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class DWConv2DTileConstraint(Conv2DTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBufferName = parseDict['data_in']
        outputBufferName = parseDict['data_out']
        weightBufferName = parseDict['weight']
        biasBufferName = parseDict['bias']

        inputBuffer = ctxt.lookup(inputBufferName)

        has_bias = False if parseDict['has_bias'] == "false" else True

        pads = parseDict["pads"]
        strides = parseDict["strides"]
        dilations = parseDict["dilations"]

        buffersOfInterest = [inputBufferName, outputBufferName, weightBufferName]
        if has_bias:
            buffersOfInterest.append(biasBufferName)

        for bufferName in buffersOfInterest:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)

        if has_bias:
            biasDimVar = tilerModel.getTensorDimVar(tensorName = biasBufferName, dimIdx = 0)

        effectiveInputHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1])) - (
            (weightHeightVar - 1) * (inputHeightVar != inputBuffer.shape[1]))
        effectiveInputWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[2])) - (
            (weightWidthVar - 1) * (inputWidthVar != inputBuffer.shape[2]))

        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(
            (outputHeightVar == (effectiveInputHeight - dilations[0] * (weightHeightVar - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint(
            (outputWidthVar == (effectiveInputWidth - dilations[1] * (weightWidthVar - 1) - 1) // strides[1] + 1))

        # DW conv: input channels == output channels per tile (not pinned to full count)
        tilerModel.addConstraint(inputChannelVar == outputChannelVar)
        tilerModel.addConstraint(weightOutChannelVar == outputChannelVar)

        if has_bias:
            tilerModel.addConstraint(biasDimVar == outputChannelVar)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        pads = parseDict["pads"]
        strides = parseDict["strides"]

        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)

        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)

        effectiveInputHeight = inputHeightVar + ((pads[0] + pads[2]) * (inputHeightVar == inputBuffer.shape[1])) - (
            (weightHeightVar - 1) * (inputHeightVar != inputBuffer.shape[1]))
        effectiveInputWidth = inputWidthVar + ((pads[1] + pads[3]) * (inputWidthVar == inputBuffer.shape[2])) - (
            (weightWidthVar - 1) * (inputWidthVar != inputBuffer.shape[2]))

        tilerModel.addConstraint(effectiveInputHeight >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(effectiveInputWidth >= parseDict['dim_kernel_y'])

        tilerModel.addConstraint((effectiveInputHeight % strides[0]) == 0)
        tilerModel.addConstraint((effectiveInputWidth % strides[1]) == 0)

        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])
        # DW weight last dim (C_in per group) is always 1 and never tiled
        tilerModel.addConstraint(weightInChannelVar == 1)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        varIn = operatorRepresentation["data_in"]
        varWeight = operatorRepresentation['weight']
        varBias = operatorRepresentation['bias']
        varOut = operatorRepresentation['data_out']

        if varBias != "NULL":
            addrNames = ['data_in', 'weight', 'bias', 'data_out']
        else:
            addrNames = ['data_in', 'weight', 'data_out']

        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        inputInCubes = []
        inputWeightCubes = []
        inputBiasCubes = []

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

        (_, weightH, weightW, _) = ctxt.lookup(varWeight).shape

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        for idx, cube in enumerate(outputCubes):
            COffset = cube.offset[3]
            (_, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = cls.computeInputCube(kernelShape = (weightH, weightW),
                                                         pads = pads,
                                                         strides = strides,
                                                         inputCSize = CSize,
                                                         outputCube = cube,
                                                         inputDims = ctxt.lookup(varIn).shape,
                                                         outputDims = ctxt.lookup(varOut).shape,
                                                         outputAbsoluteOffsets = absoluteOutputCubes[idx].absoluteOffset)

            # computeInputCube sets channel offset to 0; correct it for channel tiling
            InCube = HyperRectangle((InCube.offset[0], InCube.offset[1], InCube.offset[2], COffset), InCube.dims)

            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_in_y'].append(InCube.dims[2])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_in'].append(CSize)
            replacements['ch_im_out'].append(CSize)
            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(InCube)

            WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, weightH, weightW, 1))
            inputWeightCubes.append(WeightCube)

            if varBias != "NULL":
                BiasCube = HyperRectangle((COffset,), (CSize,))
                inputBiasCubes.append(BiasCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        if varBias == "NULL":
            for a, b in zip(inputInCubes, inputWeightCubes):
                inputLoadSchedule.append({"data_in": a, "weight": b})
        else:
            for a, b, c in zip(inputInCubes, inputWeightCubes, inputBiasCubes):
                inputLoadSchedule.append({"data_in": a, "weight": b, "bias": c})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
