# ----------------------------------------------------------------------
#
# File: NeurekaPointwiseConstraint.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t, uint32_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.Targets.Neureka.Templates.ConvTemplate import Neureka2DPWConvTemplate, getInputAddrOffset, \
    ioStridesFromDimensions
from Deeploy.Targets.PULPOpen.TileConstraints.ConvTileConstraint import Conv2DTileConstraint
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import PerformanceHint, TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme, calculateRectangleOffset


class NeurekaPWConv2DTileConstraint(TileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        # Get to-be-tiled tensor's buffers
        inputBufferName = parseDict['data_in']
        weightBufferName = parseDict['weight']
        outputBufferName = parseDict['data_out']

        # Add I/O dimensions to the model as variables
        for bufferName in [inputBufferName, weightBufferName, outputBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightInChannelMajorVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightBandwidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)  # Output Channel
        tilerModel.addConstraint(outputHeightVar == inputHeightVar)
        tilerModel.addConstraint(outputWidthVar == inputWidthVar)

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

        addrNames = ['data_in', 'weight', 'data_out']
        inputBaseOffsets, outputBaseOffsets = cls.extractBaseAddr(tilingSolution, targetMemLevel,
                                                                  operatorRepresentation, addrNames)

        varWeight = operatorRepresentation['weight']
        varOut = operatorRepresentation['data_out']

        inputInCubes = []
        inputWeightCubes = []
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

        for cube in outputCubes:
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, weightC,
                                                                          cube,
                                                                          ctxt.lookup(varOut).shape)
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

            weightShape = ctxt.lookup(varWeight).shape
            WeightCube = HyperRectangle((COffset, 0, 0), (CSize, weightShape[-2], weightShape[-1]))

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


class NeurekaRQSPWConv2DTileConstraint(NeurekaPWConv2DTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = NeurekaPWConv2DTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)

        outputBufferName = parseDict['data_out']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']

        # Add I/O dimensions to the model as variables
        for bufferName in [mulBufferName, addBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)
        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        variableReplacementSchedule, tilingSchedule = super().serializeTilingSolution(
            tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation)

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['mul', 'add']
        inputRequantBaseOffsets, _ = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                                         addrNames)
        newInputBaseOffsets = {**tilingSchedule.inputBaseOffsets, **inputRequantBaseOffsets}

        inputRequantCubes = []
        for cube in outputCubes:
            (_, _, _, COffset) = cube.offset
            (_, _, _, CSize) = cube.dims
            inputRequantCubes.append(HyperRectangle((COffset,), (CSize,)))
        newInputLoadSchedule = [{
            **schedule, "add": requant,
            "mul": requant
        } for schedule, requant in zip(tilingSchedule.inputLoadSchedule, inputRequantCubes)]

        newTilingSchedule = TilingSchedule(newInputBaseOffsets, tilingSchedule.outputBaseOffsets, newInputLoadSchedule,
                                           tilingSchedule.outputLoadSchedule)

        return variableReplacementSchedule, newTilingSchedule


class NeurekaWmemPWConv2DTileConstraint(TileConstraint):

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

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)
        tilerModel.addConstraint(outputHeightVar == inputHeightVar)
        tilerModel.addConstraint(outputWidthVar == inputWidthVar)

        # Don't tile weights in weight memory
        tilerModel.addConstraint(weightOutChannelVar == weightOutChannelVar.Max())

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
            "weight_addr_offset": [],
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
            "weight_addr_offset": PointerClass(uint32_t),
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

        for absoluteCube in absoluteOutputCubes:
            cube = absoluteCube.rectangle
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, weightC,
                                                                          cube,
                                                                          ctxt.lookup(varOut).shape)
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

            _, _, _, absoluteCOffset = absoluteCube.absoluteOffset
            weightShape = ctxt.lookup(varWeight).shape
            WeightCube = HyperRectangle((absoluteCOffset, 0, 0), (CSize, weightShape[-2], weightShape[-1]))
            replacements['weight_addr_offset'].append(calculateRectangleOffset(WeightCube, ctxt.lookup(varWeight)))

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a in inputInCubes:
            inputLoadSchedule.append({"data_in": a})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule


class NeurekaWmemRQSPWConv2DTileConstraint(NeurekaWmemPWConv2DTileConstraint):

    @staticmethod
    def addGeometricalConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:
        tilerModel = NeurekaWmemPWConv2DTileConstraint.addGeometricalConstraint(tilerModel, parseDict, ctxt)

        outputBufferName = parseDict['data_out']
        mulBufferName = parseDict['mul']
        addBufferName = parseDict['add']

        # Add I/O dimensions to the model as variables
        for bufferName in [mulBufferName, addBufferName]:
            tilerModel.addTensorDimToModel(ctxt, bufferName)

        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)
        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        return tilerModel

    @classmethod
    def serializeTilingSolution(
            cls, tilingSolution: NodeMemoryConstraint, absoluteOutputCubes: List[AbsoluteHyperRectangle],
            targetMemLevel: str, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> Tuple[VariableReplacementScheme, TilingSchedule]:
        variableReplacementSchedule, tilingSchedule = super().serializeTilingSolution(
            tilingSolution, absoluteOutputCubes, targetMemLevel, ctxt, operatorRepresentation)

        outputCubes = [cube.rectangle for cube in absoluteOutputCubes]

        addrNames = ['mul', 'add']
        inputRequantBaseOffsets, _ = cls.extractBaseAddr(tilingSolution, targetMemLevel, operatorRepresentation,
                                                         addrNames)
        newInputBaseOffsets = {**tilingSchedule.inputBaseOffsets, **inputRequantBaseOffsets}

        inputRequantCubes = []
        for cube in outputCubes:
            (_, _, _, COffset) = cube.offset
            (_, _, _, CSize) = cube.dims
            inputRequantCubes.append(HyperRectangle((COffset,), (CSize,)))
        newInputLoadSchedule = [{
            **schedule, "add": requant,
            "mul": requant
        } for schedule, requant in zip(tilingSchedule.inputLoadSchedule, inputRequantCubes)]

        newTilingSchedule = TilingSchedule(newInputBaseOffsets, tilingSchedule.outputBaseOffsets, newInputLoadSchedule,
                                           tilingSchedule.outputLoadSchedule)

        return variableReplacementSchedule, newTilingSchedule
