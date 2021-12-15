# ----------------------------------------------------------------------
#
# File: ConvTemplate.py
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

from abc import abstractmethod
from typing import Dict, List, Tuple

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, NodeTemplate, OperatorRepresentation


def _getNumTiles(fullDim: int, tileDim: int) -> int:
    return int(np.ceil(fullDim / tileDim))


def _getBorderTileSize(fullDim: int, tileDim: int) -> int:
    return fullDim % tileDim if fullDim % tileDim > 0 else tileDim


def ioStridesFromDimensions(width: int, channel: int, bits: int) -> Tuple[int, int]:
    """stridesFromDimensions
    Returns strides in bytes.
    """
    width_stride = channel * bits // 8
    height_stride = width * width_stride
    return height_stride, width_stride


def getNormQuantConf0(use_relu: bool, layerwise_output_shift: int, scale_bits: int, use_bias: bool,
                      use_shift: bool) -> int:
    conf0 = 0
    conf0 |= 1 << 4  # Use Normalization and quantization
    if scale_bits == 32:
        conf0 |= 2 << 12
    conf0 |= layerwise_output_shift << 16
    if not use_relu:
        conf0 |= 1 << 23
    if use_shift:
        conf0 |= 1 << 24
    if use_bias:
        conf0 |= 1 << 25
    return conf0


def getInputAddrOffset(width_in: int, width_in_stride: int, padding_top: int, padding_left: int) -> int:
    return (padding_top * width_in + padding_left) * width_in_stride


class NeurekaConvTemplate(NodeTemplate):

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @classmethod
    @abstractmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        pass

    @classmethod
    @abstractmethod
    def getWeightStrides(cls, channel_in: int) -> Tuple[int, int, int]:
        pass

    @classmethod
    @abstractmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        pass

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        data_in: ConstantBuffer = ctxt.lookup(operatorRepresentation['data_in'])
        data_out: ConstantBuffer = ctxt.lookup(operatorRepresentation['data_out'])
        weight: ConstantBuffer = ctxt.lookup(operatorRepresentation['weight'])

        operatorRepresentation['input_signed'] = data_in._type.referencedType.typeMin < 0
        operatorRepresentation['use_relu'] = data_out._type.referencedType.typeMin >= 0

        operatorRepresentation['input_bits'] = data_in._type.referencedType.typeWidth
        operatorRepresentation['output_bits'] = data_out._type.referencedType.typeWidth
        operatorRepresentation['weight_bits'] = weight._type.referencedType.typeWidth

        operatorRepresentation["input_typeWidth_bytes"] = int(np.ceil(data_in._type.referencedType.typeWidth / 8))
        operatorRepresentation["output_typeWidth_bytes"] = int(np.ceil(data_out._type.referencedType.typeWidth / 8))

        operatorRepresentation["weight_addr_offset"] = 0

        operatorRepresentation["use_wmem"] = hasattr(weight,
                                                     "_memoryLevel") and weight._memoryLevel == "WeightMemory_SRAM"

        dim_im_in_x_stride, dim_im_in_y_stride = ioStridesFromDimensions(operatorRepresentation["dim_im_in_y"],
                                                                         operatorRepresentation["ch_im_in"],
                                                                         operatorRepresentation["input_bits"])
        operatorRepresentation["dim_im_in_y_stride"] = dim_im_in_y_stride
        operatorRepresentation["dim_im_in_x_stride"] = dim_im_in_x_stride

        dim_im_out_x_stride, dim_im_out_y_stride = ioStridesFromDimensions(operatorRepresentation["dim_im_out_y"],
                                                                           operatorRepresentation["ch_im_out"],
                                                                           operatorRepresentation["output_bits"])
        operatorRepresentation["dim_im_out_y_stride"] = dim_im_out_y_stride
        operatorRepresentation["dim_im_out_x_stride"] = dim_im_out_x_stride

        operatorRepresentation["input_addr_offset"] = getInputAddrOffset(operatorRepresentation["dim_im_in_y"],
                                                                         operatorRepresentation["dim_im_in_y_stride"],
                                                                         operatorRepresentation["padding_y_top"],
                                                                         operatorRepresentation["padding_x_left"])

        nKo, nKi, nHo, nWo, bKo, bKi, bHo, bWo, bHi, bWi = self.getCounters(
            operatorRepresentation["ch_im_in"], operatorRepresentation["dim_im_out_x"],
            operatorRepresentation["dim_im_out_y"], operatorRepresentation["ch_im_out"],
            operatorRepresentation["padding_y_bottom"], operatorRepresentation["padding_x_right"],
            operatorRepresentation)

        operatorRepresentation["nKo"] = nKo
        operatorRepresentation["nKi"] = nKi
        operatorRepresentation["nHo"] = nHo
        operatorRepresentation["nWo"] = nWo
        operatorRepresentation["bKo"] = bKo
        operatorRepresentation["bKi"] = bKi
        operatorRepresentation["bHo"] = bHo
        operatorRepresentation["bWo"] = bWo
        operatorRepresentation["bHi"] = bHi
        operatorRepresentation["bWi"] = bWi

        weightStrideD0, weightStrideD1, weightStrideD2 = self.getWeightStrides(operatorRepresentation["ch_im_in"])

        operatorRepresentation["weightStrideD0"] = weightStrideD0
        operatorRepresentation["weightStrideD1"] = weightStrideD1
        operatorRepresentation["weightStrideD2"] = weightStrideD2

        operatorRepresentation["conf0"] = self.getConf0(operatorRepresentation["output_bits"],
                                                        operatorRepresentation["weight_bits"],
                                                        operatorRepresentation["input_signed"],
                                                        operatorRepresentation["use_wmem"])

        operatorRepresentation["wmem_addr_offset"] = 0x10400000 if operatorRepresentation["use_wmem"] else 0

        # If requantized
        if operatorRepresentation["mul"] != "NULL":
            mulBuff = ctxt.lookup(operatorRepresentation["mul"])
            mulBits = mulBuff._type.referencedType.typeWidth
            operatorRepresentation["conf0"] |= getNormQuantConf0(operatorRepresentation["use_relu"],
                                                                 operatorRepresentation["log2D"], mulBits, "add"
                                                                 in operatorRepresentation, False)
        return ctxt, operatorRepresentation, []


class Neureka2DPWConvTemplate(NeurekaConvTemplate):

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @classmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        n_channel_out_subtiles = _getNumTiles(channel_out, 32)
        n_channel_in_subtiles = _getNumTiles(channel_in, 32)
        n_height_out_subtiles = _getNumTiles(height_out, 6)
        n_width_out_subtiles = _getNumTiles(width_out, 6)

        channel_out_border = _getBorderTileSize(channel_out, 32)
        channel_in_border = _getBorderTileSize(channel_in, 32)
        height_out_border = _getBorderTileSize(height_out, 6)
        width_out_border = _getBorderTileSize(width_out, 6)
        height_in_border = height_out_border - padding_bottom
        width_in_border = width_out_border - padding_right

        return (n_channel_out_subtiles, n_channel_in_subtiles, n_height_out_subtiles, n_width_out_subtiles,
                channel_out_border, channel_in_border, height_out_border, width_out_border, height_in_border,
                width_in_border)

    @classmethod
    def getWeightStrides(cls, channel_in: int) -> Tuple[int, int, int]:
        n_channel_in = _getNumTiles(channel_in, 32)
        _NEUREKA_WEIGHT_BANDWIDTH_BYTES = 32
        return _NEUREKA_WEIGHT_BANDWIDTH_BYTES, _NEUREKA_WEIGHT_BANDWIDTH_BYTES * n_channel_in, 0

    @classmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        conf0 = 0
        conf0 |= weight_bits - 1
        conf0 |= 2 << 5  # PW MODE
        if use_wmem:
            conf0 |= 1 << 9
        conf0 |= 1 << 15  # Layerwise weight offset mode
        if output_bits == 32:
            conf0 |= 2 << 21
        if input_signed:
            conf0 |= 1 << 26
        return conf0


class Neureka2DDWConvTemplate(NeurekaConvTemplate):

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @classmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        _ = operatorRepresentation  # operatorRepresentation not accessed for now because it's just for pointwise kernels

        n_channel_out_subtiles = _getNumTiles(channel_out, 28)
        n_channel_in_subtiles = n_channel_out_subtiles
        n_height_out_subtiles = _getNumTiles(height_out, 6)
        n_width_out_subtiles = _getNumTiles(width_out, 6)

        channel_out_border = _getBorderTileSize(channel_out, 28)
        channel_in_border = channel_out_border
        height_out_border = _getBorderTileSize(height_out, 6)
        width_out_border = _getBorderTileSize(width_out, 6)
        height_in_border = height_out_border + 2 - padding_bottom
        width_in_border = width_out_border + 2 - padding_right

        return (n_channel_out_subtiles, n_channel_in_subtiles, n_height_out_subtiles, n_width_out_subtiles,
                channel_out_border, channel_in_border, height_out_border, width_out_border, height_in_border,
                width_in_border)

    @classmethod
    def getWeightStrides(cls, channel_in: int) -> Tuple[int, int, int]:
        n_channel_in = _getNumTiles(channel_in, 28)
        _NEUREKA_WEIGHT_BANDWIDTH_BYTES = 32
        return _NEUREKA_WEIGHT_BANDWIDTH_BYTES, 0, 0

    @classmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        conf0 = 0
        conf0 |= weight_bits - 1
        conf0 |= 1 << 5  # DW MODE
        if use_wmem:
            conf0 |= 1 << 9
        conf0 |= 1 << 15  # Layerwise weight offset mode
        if output_bits == 32:
            conf0 |= 2 << 21
        if input_signed:
            conf0 |= 1 << 26
        return conf0


class Neureka2DDenseConvTemplate(NeurekaConvTemplate):

    def __init__(self, templateStr: str):
        super().__init__(templateStr)

    @classmethod
    def getCounters(
            cls, channel_in: int, height_out: int, width_out: int, channel_out: int, padding_bottom: int,
            padding_right: int,
            operatorRepresentation: OperatorRepresentation) -> Tuple[int, int, int, int, int, int, int, int, int, int]:
        _ = operatorRepresentation  # operatorRepresentation not accessed for now because it's just for pointwise kernels

        n_channel_out_subtiles = _getNumTiles(channel_out, 28)
        n_channel_in_subtiles = _getNumTiles(channel_in, 28)
        n_height_out_subtiles = _getNumTiles(height_out, 6)
        n_width_out_subtiles = _getNumTiles(width_out, 6)

        channel_out_border = _getBorderTileSize(channel_out, 28)
        channel_in_border = _getBorderTileSize(channel_in, 28)
        height_out_border = _getBorderTileSize(height_out, 6)
        width_out_border = _getBorderTileSize(width_out, 6)
        height_in_border = height_out_border + 2 - padding_bottom
        width_in_border = width_out_border + 2 - padding_right

        return (n_channel_out_subtiles, n_channel_in_subtiles, n_height_out_subtiles, n_width_out_subtiles,
                channel_out_border, channel_in_border, height_out_border, width_out_border, height_in_border,
                width_in_border)

    @classmethod
    def getWeightStrides(cls, channel_in: int) -> Tuple[int, int, int]:
        n_channel_in = _getNumTiles(channel_in, 28)
        _NEUREKA_WEIGHT_BANDWIDTH_BYTES = 32
        return _NEUREKA_WEIGHT_BANDWIDTH_BYTES, _NEUREKA_WEIGHT_BANDWIDTH_BYTES * 8 * n_channel_in, 0

    @classmethod
    def getConf0(cls, output_bits: int, weight_bits: int, input_signed: bool, use_wmem: bool) -> int:
        conf0 = 0
        conf0 |= weight_bits - 1
        if use_wmem:
            conf0 |= 1 << 9
        conf0 |= 1 << 15  # Layerwise weight offset mode
        if output_bits == 32:
            conf0 |= 2 << 21
        if input_signed:
            conf0 |= 1 << 26
        return conf0


NeurekaTaskInitTemplateStr = """
// N-EUREKA Task Init
neureka_task_t task = {
    .data = (neureka_task_data_t) {
        .weights_addr = (uint32_t)${weight} - ${wmem_addr_offset} + ${weight_addr_offset},
        .infeat_addr = (uint32_t)${data_in} - ${input_addr_offset},
        .outfeat_addr = (uint32_t)${data_out},
        .scale_addr = (uint32_t)${mul},
        .scale_shift_addr = (uint32_t)${shift},
        .scale_bias_addr = (uint32_t)${add},
        .cfg = (neureka_cfg_t) {
            .input_stride = (neureka_stride_t) {
                .d0 = ${dim_im_in_y_stride},
                .d1 = ${dim_im_in_x_stride},
                .d2 = 0
            },
            .output_stride = (neureka_stride_t) {
                .d0 = NEUREKA_OUTPUT_BANDWIDTH_BYTES,
                .d1 = ${dim_im_out_y_stride},
                .d2 = ${dim_im_out_x_stride}
            },
            task.data.cfg.weights_stride = (neureka_stride_t) {
                .d0 = ${weightStrideD0},
                .d1 = ${weightStrideD1},
                .d2 = ${weightStrideD2}
            },
            .subtile = (neureka_subtile_t) {
                .number = {
                    .KoKi = nnx_concat_half(${nKo}, ${nKi}),
                    .HoWo = nnx_concat_half(${nHo}, ${nWo})
                },
                .remainder = {
                    .KoKi = nnx_concat_half(${bKo}, ${bKi}),
                    .HoWo = nnx_concat_half(${bHo}, ${bWo}),
                    .HiWi = nnx_concat_half(${bHi}, ${bWi})
                }
            },
            .padding = (${padding_y_top} << 28) + (${padding_x_right} << 24) + (${padding_y_bottom} << 20) + (${padding_x_left} << 16),
            .weight_offset_factor = ${weight_offset},
            .filter_mask = 0,
            .conf0 = ${conf0},
        }
    }
};
"""

NeurekaTaskExecutionTemplateStr = """
// N-EUREKA Task Execution
neureka_nnx_dispatch_wait(neureka_siracusa_get_dev());
neureka_nnx_dispatch(neureka_siracusa_get_dev(), &task);
neureka_nnx_resolve_wait(neureka_siracusa_get_dev(), &task);
"""

NeurekaRqntPWConv2D_Template = Neureka2DPWConvTemplate(NeurekaTaskInitTemplateStr + NeurekaTaskExecutionTemplateStr)
NeurekaPWConv2D_Template = Neureka2DPWConvTemplate(NeurekaTaskInitTemplateStr + NeurekaTaskExecutionTemplateStr)

NeurekaRqntDWConv2D_Template = Neureka2DDWConvTemplate(NeurekaTaskInitTemplateStr + NeurekaTaskExecutionTemplateStr)
NeurekaDWConv2D_Template = Neureka2DDWConvTemplate(NeurekaTaskInitTemplateStr + NeurekaTaskExecutionTemplateStr)

NeurekaRqntDenseConv2D_Template = Neureka2DDenseConvTemplate(NeurekaTaskInitTemplateStr +
                                                             NeurekaTaskExecutionTemplateStr)
NeurekaDenseConv2D_Template = Neureka2DDenseConvTemplate(NeurekaTaskInitTemplateStr + NeurekaTaskExecutionTemplateStr)
