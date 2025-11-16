# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation, VariableBuffer
from Deeploy.Targets.Neureka.Templates.ConvTemplate import Neureka2DPWConvTemplate, getInputAddrOffset, \
    ioStridesFromDimensions
from Deeploy.Targets.Neureka.TileConstraints.RequantHelpers import requantAddGeometricalConstraint, requantLoadSchedule
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme, calculateFlatOffsetInBytes


class NeurekaPWConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        outputBufferName = parseDict['data_out']

        for bufferName in [inputBufferName, weightBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputHeightVar == inputHeightVar)
        tilerModel.addConstraint(outputWidthVar == inputWidthVar)

        weightBuffer = ctxt.lookup(weightBufferName)
        if hasattr(weightBuffer, "_memoryLevel") and weightBuffer._memoryLevel == "WeightMemory_SRAM":
            tilerModel.addConstraint(weightOutChannelVar == weightOutChannelVar.Max())
        else:
            tilerModel.addConstraint(weightOutChannelVar == outputChannelVar)

        tilerModel.addConstraint(inputHeightVar >= 1)
        tilerModel.addConstraint(inputWidthVar >= 1)

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])
        outputBuffer = ctxt.lookup(name = parseDict['data_out'])

        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)
        weightInChannelMajorVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightBandwidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)

        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBuffer.name, dimIdx = 3)

        strides = parseDict["strides"]
        padding = parseDict["pads"]

        # LMACAN: Force full input channel to avoid partial results
        tilerModel.addConstraint(inputChannelVar == inputChannelVar.Max())
        tilerModel.addConstraint(weightInChannelMajorVar == weightInChannelMajorVar.Max())
        tilerModel.addConstraint(weightBandwidthVar == weightBandwidthVar.Max())

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        # N-EUREKA tile constraints to align with N-EUREKA's hardware subtiling
        if parseDict["dim_im_out_x"] > 6:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "dim_im_out_x",
                                                      outputHeightVar,
                                                      6,
                                                      strategy = PerformanceHint(priority = 3))
        else:
            tilerModel.addConstraint(outputHeightVar == outputHeightVar.Max(), strategy = PerformanceHint(priority = 3))

        if parseDict["dim_im_out_y"] > 6:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "dim_im_out_y",
                                                      outputWidthVar,
                                                      6,
                                                      strategy = PerformanceHint(priority = 2))
        else:
            tilerModel.addConstraint(outputWidthVar == outputWidthVar.Max(), strategy = PerformanceHint(priority = 2))

        if parseDict["ch_im_out"] > 32:
            tilerModel.addTileSizeDivisibleConstraint(parseDict,
                                                      "ch_im_out",
                                                      outputChannelVar,
                                                      32,
                                                      strategy = PerformanceHint(priority = 1))
        else:
            tilerModel.addConstraint(outputChannelVar == outputChannelVar.Max(),
                                     strategy = PerformanceHint(priority = 1))

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

        varWeight = operatorRepresentation['weight']
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        replacements: Dict[str, List[int]] = {
            "padding_y_top": [],
            "padding_y_bottom": [],
            "padding_x_left": [],
            "padding_x_right": [],
            "dim_im_in_x_stride": [],
            "dim_im_in_y_stride": [],
            "dim_im_out_x_stride": [],
            "dim_im_out_y_stride": [],
            "input_addr_offset": [],
            "nKo": [],
            "nKi": [],
            "nHo": [],
            "nWo": [],
            "bKo": [],
            "bKi": [],
            "bHo": [],
            "bWo": [],
            "bHi": [],
            "bWi": [],
        }

        replacementTypes = {
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t),
            "dim_im_in_x_stride": PointerClass(uint32_t),
            "dim_im_in_y_stride": PointerClass(uint32_t),
            "dim_im_out_x_stride": PointerClass(uint32_t),
            "dim_im_out_y_stride": PointerClass(uint32_t),
            "input_addr_offset": PointerClass(uint32_t),
            "nKo": PointerClass(uint16_t),
            "nKi": PointerClass(uint16_t),
            "nHo": PointerClass(uint16_t),
            "nWo": PointerClass(uint16_t),
            "bKo": PointerClass(uint16_t),
            "bKi": PointerClass(uint16_t),
            "bHo": PointerClass(uint16_t),
            "bWo": PointerClass(uint16_t),
            "bHi": PointerClass(uint16_t),
            "bWi": PointerClass(uint16_t),
        }

        weightH = operatorRepresentation['dim_kernel_y']
        weightW = operatorRepresentation['dim_kernel_x']
        weightC = operatorRepresentation['ch_im_in']

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        outputBuffer = ctxt.lookup(varOut)
        assert isinstance(outputBuffer, VariableBuffer)

        for cube in outputCubes:
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, weightC,
                                                                          cube, outputBuffer.shape)
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inBSize, inHSize, inWSize, inCSize = InCube.dims

            dim_im_in_x_stride, dim_im_in_y_stride = ioStridesFromDimensions(inWSize, inCSize,
                                                                             operatorRepresentation["input_bits"])
            replacements['dim_im_in_x_stride'].append(dim_im_in_x_stride)
            replacements['dim_im_in_y_stride'].append(dim_im_in_y_stride)
            dim_im_out_x_stride, dim_im_out_y_stride = ioStridesFromDimensions(WSize, CSize,
                                                                               operatorRepresentation["output_bits"])
            replacements['dim_im_out_x_stride'].append(dim_im_out_x_stride)
            replacements['dim_im_out_y_stride'].append(dim_im_out_y_stride)

            replacements['input_addr_offset'].append(
                getInputAddrOffset(inWSize, dim_im_in_y_stride, padding_top, padding_left))

            nKo, nKi, nHo, nWo, bKo, bKi, bHo, bWo, bHi, bWi = Neureka2DPWConvTemplate.getCounters(
                inCSize, HSize, WSize, CSize, padding_bottom, padding_right, operatorRepresentation)

            replacements["nKo"].append(nKo)
            replacements["nKi"].append(nKi)
            replacements["nHo"].append(nHo)
            replacements["nWo"].append(nWo)
            replacements["bKo"].append(bKo)
            replacements["bKi"].append(bKi)
            replacements["bHo"].append(bHo)
            replacements["bWo"].append(bWo)
            replacements["bHi"].append(bHi)
            replacements["bWi"].append(bWi)

            inputInCubes.append(InCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a in inputInCubes:
            inputLoadSchedule.append({"data_in": a})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        weightBuffer = ctxt.lookup(varWeight)
        assert isinstance(weightBuffer, VariableBuffer)
        weightShape = weightBuffer.shape

        if hasattr(weightBuffer, "_memoryLevel") and weightBuffer._memoryLevel == "WeightMemory_SRAM":
            replacements['weight_addr_offset'] = []
            replacementTypes['weight_addr_offset'] = PointerClass(uint32_t)
            for absoluteCube in absoluteOutputCubes:
                COffset, CSize = absoluteCube.absoluteOffset[-1], absoluteCube.rectangle.dims[-1]
                WeightCube = HyperRectangle((COffset, 0, 0), (CSize, weightShape[-2], weightShape[-1]))
                replacements['weight_addr_offset'].append(calculateFlatOffsetInBytes(WeightCube, weightBuffer))
        else:
            inputWeightBaseOffsets, outputWeightBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                                  operatorRepresentation, ['weight'])
            inputBaseOffsets.update(inputWeightBaseOffsets)
            outputBaseOffsets.update(outputWeightBaseOffsets)

            for cube, load in zip(outputCubes, inputLoadSchedule):
                COffset, CSize = cube.offset[-1], cube.dims[-1]
                load['weight'] = HyperRectangle((COffset, 0, 0), (CSize, weightShape[-2], weightShape[-1]))

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class NeurekaRQSPWConv2DTileConstraint(NeurekaPWConv2DTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = NeurekaPWConv2DTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)
        return requantAddGeometricalConstraint(tilerModel, parseDict, ctxt)

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        variableReplacementSchedule, tilingSchedule = super().serializeTilingSolution(
            tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation)

        addrNames = ['mul', 'add']
        inputRequantBaseOffsets, _ = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                                         addrNames)
        newInputBaseOffsets = {**tilingSchedule.inputBaseOffsets, **inputRequantBaseOffsets}

        requantSchedule = requantLoadSchedule(absoluteOutputCubes, ctxt, operatorRepresentation)
        newInputLoadSchedule = [{
            **load,
            **rqLoad
        } for load, rqLoad in zip(tilingSchedule.inputLoadSchedule, requantSchedule)]

        newTilingSchedule = TilingSchedule(newInputBaseOffsets, tilingSchedule.outputBaseOffsets, newInputLoadSchedule,
                                           tilingSchedule.outputLoadSchedule)

        return variableReplacementSchedule, newTilingSchedule
