# ----------------------------------------------------------------------
#
# File: ConvTileConstraint.py
#
# Last edited: 09.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
# - Moritz Scherer, scheremo@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import uint8_t, uint16_t
from Deeploy.DeeployTypes import NetworkContext, OperatorRepresentation
from Deeploy.TilingExtension.MemoryConstraints import NodeMemoryConstraint
from Deeploy.TilingExtension.TileConstraint import TileConstraint
from Deeploy.TilingExtension.TilerModel import TilerModel
from Deeploy.TilingExtension.TilingCodegen import AbsoluteHyperRectangle, HyperRectangle, TilingSchedule, \
    VariableReplacementScheme


class Conv2DTileConstraint(TileConstraint):

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

        inputBatchVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 0)
        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBufferName, dimIdx = 3)

        weightOutChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBufferName, dimIdx = 3)

        outputBatchVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 0)
        outputHeightVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 1)
        outputWidthVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 2)
        outputChannelVar = tilerModel.getTensorDimVar(tensorName = outputBufferName, dimIdx = 3)

        addChannelVar = tilerModel.getTensorDimVar(tensorName = addBufferName, dimIdx = 0)
        mulChannelVar = tilerModel.getTensorDimVar(tensorName = mulBufferName, dimIdx = 0)

        # Map output dims to inputs dims
        tilerModel.addConstraint(outputBatchVar == inputBatchVar)  # Batch
        tilerModel.addConstraint(outputChannelVar == weightOutChannelVar)  # Output Channel

        tilerModel.addConstraint(outputChannelVar == addChannelVar)
        tilerModel.addConstraint(outputChannelVar == mulChannelVar)

        inputBuffer = ctxt.lookup(inputBufferName)

        effectiveHeight = inputHeightVar + ((padding[0] + padding[2]) * (inputHeightVar == inputBuffer.shape[1]))
        effectiveWidth = inputWidthVar + ((padding[1] + padding[3]) * (inputWidthVar == inputBuffer.shape[2]))

        tilerModel.addConstraint((outputHeightVar == (effectiveHeight - (weightHeightVar - 1) - 1) // strides[0] + 1))
        tilerModel.addConstraint((outputWidthVar == (effectiveWidth - (weightWidthVar - 1) - 1) // strides[1] + 1))

        return tilerModel

    @staticmethod
    def addPolicyConstraint(tilerModel: TilerModel, parseDict: Dict, ctxt: NetworkContext) -> TilerModel:

        # Get to-be-tiled tensor's buffers
        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        inputHeightVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 1)
        inputWidthVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 2)
        inputChannelVar = tilerModel.getTensorDimVar(tensorName = inputBuffer.name, dimIdx = 3)

        outputChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 0)
        weightHeightVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 1)
        weightWidthVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 2)
        weightInChannelVar = tilerModel.getTensorDimVar(tensorName = weightBuffer.name, dimIdx = 3)

        strides = parseDict["strides"]
        padding = parseDict["pads"]

        # VIC: Force at least one row of A and one col of B in the GEMM (since it's a im2col Conv) to avoid partial results
        tilerModel.addConstraint(inputChannelVar == parseDict['ch_im_in'])

        if (parseDict["ch_im_out"] >= 8):
            tilerModel.addMinTileSizeConstraint(parseDict, 'ch_im_out', outputChannelVar, 8)

        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_y'])
        tilerModel.addConstraint(weightInChannelVar == parseDict['ch_im_in'])

        # VIC: Constraint the minimum tile size such that we can apply at least one kernel on it
        tilerModel.addConstraint(inputHeightVar >= parseDict['dim_kernel_x'])
        tilerModel.addConstraint(inputWidthVar >= parseDict['dim_kernel_y'])

        tilerModel.addConstraint(weightHeightVar == parseDict['dim_kernel_x'])
        tilerModel.addConstraint(weightWidthVar == parseDict['dim_kernel_y'])

        tilerModel.addConstraint((inputHeightVar % strides[0]) == 0)
        tilerModel.addConstraint((inputWidthVar % strides[1]) == 0)

        return tilerModel

    @staticmethod
    def constructSymbolicNodeRep(tilerModel: TilerModel, parseDict: Dict,
                                 ctxt: NetworkContext) -> Dict[str, Union[int, IntVar]]:

        inputBuffer = ctxt.lookup(name = parseDict['data_in'])
        weightBuffer = ctxt.lookup(name = parseDict['weight'])

        symbolicParseDict = parseDict.copy()
        symbolicParseDict['dim_im_in_x'] = tilerModel.getTensorDimVar(inputBuffer.name, 1)
        symbolicParseDict['dim_kernel_x'] = tilerModel.getTensorDimVar(weightBuffer.name, 1)
        symbolicParseDict['dim_kernel_y'] = tilerModel.getTensorDimVar(weightBuffer.name, 2)

        return symbolicParseDict

    @staticmethod
    def computeMargins(kernelShape: Tuple[int, ...]) -> Tuple[int, ...]:
        if kernelShape[1] % 2 == 0:
            leftMargin = 0
            rightMargin = 0
        else:
            leftMargin = ((kernelShape[1]) // 2)
            rightMargin = ((kernelShape[1]) // 2)

        if kernelShape[0] % 2 == 0:
            topMargin = 0
            bottomMargin = 0
        else:
            topMargin = ((kernelShape[0]) // 2)
            bottomMargin = ((kernelShape[0]) // 2)

        return leftMargin, rightMargin, topMargin, bottomMargin

    @staticmethod
    def computeInputCube(kernelShape: Tuple[int, ...], pads: Tuple[int, ...], strides: Tuple[int, ...],
                         weightChannels: int, outputCube: HyperRectangle,
                         outputDims: Tuple[int, ...]) -> Tuple[HyperRectangle, Tuple[int, ...]]:

        (BatchOffset, HOffset, WOffset, COffset) = outputCube.offset
        (BatchSize, HSize, WSize, CSize) = outputCube.dims

        leftMargin, rightMargin, topMargin, bottomMargin = Conv2DTileConstraint.computeMargins(kernelShape)

        padding_top = (HOffset == 0) * pads[0]
        padding_bottom = (HOffset + HSize == outputDims[1]) * pads[2]

        padding_left = (WOffset == 0) * pads[1]
        padding_right = (WOffset + WSize == outputDims[2]) * pads[3]

        inputHOffset = HOffset * strides[0] - topMargin * (HOffset != 0)
        inputWOffset = WOffset * strides[1] - leftMargin * (WOffset != 0)

        inputHSize = HSize * strides[0] + (topMargin + bottomMargin) - (padding_top + padding_bottom)
        inputWSize = WSize * strides[1] + (leftMargin + rightMargin) - (padding_left + padding_right)

        InCube = HyperRectangle((BatchOffset, inputHOffset, inputWOffset, 0),
                                (BatchSize, inputHSize, inputWSize, weightChannels))

        return InCube, (padding_left, padding_right, padding_top, padding_bottom)

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
            "padding_y_top": PointerClass(uint8_t),
            "padding_y_bottom": PointerClass(uint8_t),
            "padding_x_left": PointerClass(uint8_t),
            "padding_x_right": PointerClass(uint8_t)
        }

        weightH = ctxt.lookup(varWeight).shape[1]
        weightW = ctxt.lookup(varWeight).shape[2]
        weightC = ctxt.lookup(varWeight).shape[3]

        pads = operatorRepresentation['pads']
        strides = operatorRepresentation['strides']

        for cube in outputCubes:
            (BatchOffset, HOffset, WOffset, COffset) = cube.offset
            (BatchSize, HSize, WSize, CSize) = cube.dims

            InCube, padding_tuple = Conv2DTileConstraint.computeInputCube((weightH, weightW), pads, strides, weightC,
                                                                          cube,
                                                                          ctxt.lookup(varOut).shape)
            padding_left, padding_right, padding_top, padding_bottom = padding_tuple

            replacements['dim_im_in_x'].append(InCube.dims[1])
            replacements['dim_im_in_y'].append(InCube.dims[2])
            replacements['dim_im_out_x'].append(HSize)
            replacements['dim_im_out_y'].append(WSize)
            replacements['ch_im_out'].append(CSize)

            replacements['padding_y_top'].append(padding_top)
            replacements['padding_y_bottom'].append(padding_bottom)
            replacements['padding_x_left'].append(padding_left)
            replacements['padding_x_right'].append(padding_right)

            inputInCubes.append(InCube)

            RequantCube = HyperRectangle((COffset,), (CSize,))
            WeightCube = HyperRectangle((COffset, 0, 0, 0), (CSize, weightH, weightW, weightC))

            inputWeightCubes.append(WeightCube)
            inputAddCubes.append(RequantCube)
            inputMulCubes.append(RequantCube)

        inputLoadSchedule = []
        outputLoadSchedule = []

        for a, b, add, mul in zip(inputInCubes, inputWeightCubes, inputAddCubes, inputMulCubes):
            inputLoadSchedule.append({"data_in": a, "weight": b, "add": add, "mul": mul})

        for out in outputCubes:
            outputLoadSchedule.append({"data_out": out})

        tilingSchedule = TilingSchedule(inputBaseOffsets, outputBaseOffsets, inputLoadSchedule, outputLoadSchedule)
        variableReplacementSchedule = VariableReplacementScheme(replacements, replacementTypes)

        return variableReplacementSchedule, tilingSchedule
