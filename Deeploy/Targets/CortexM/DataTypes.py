# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass, Struct, VoidType
from Deeploy.CommonExtensions.DataTypes import int32_t


class cmsis_nn_context(Struct):
    typeName = "cmsis_nn_context"
    structTypeDict = {"buf": PointerClass(VoidType), "size": int32_t}


class cmsis_nn_tile(Struct):
    typeName = "cmsis_nn_tile"
    structTypeDict = {"w": int32_t, "h": int32_t}


class cmsis_nn_activation(Struct):
    typeName = "cmsis_nn_activation"
    structTypeDict = {"min": int32_t, "max": int32_t}


class cmsis_nn_dims(Struct):
    typeName = "cmsis_nn_dims"
    structTypeDict = {"n": int32_t, "h": int32_t, "w": int32_t, "c": int32_t}


class cmsis_nn_per_channel_quant_params(Struct):
    typeName = "cmsis_nn_per_channel_quant_params"
    structTypeDict = {"multiplier": PointerClass(int32_t), "shift": PointerClass(int32_t)}


class cmsis_nn_per_tensor_quant_params(Struct):
    typeName = "cmsis_nn_per_tensor_quant_params"
    structTypeDict = {"multiplier": int32_t, "shift": int32_t}


class cmsis_nn_conv_params(Struct):
    typeName = "cmsis_nn_conv_params"
    structTypeDict = {
        "input_offset": int32_t,
        "output_offset": int32_t,
        "stride": cmsis_nn_tile,
        "padding": cmsis_nn_tile,
        "dilation": cmsis_nn_tile,
        "activation": cmsis_nn_activation
    }


class cmsis_nn_fc_params(Struct):
    typeName = "cmsis_nn_fc_params"
    structTypeDict = {
        "input_offset": int32_t,
        "filter_offset": int32_t,
        "output_offset": int32_t,
        "activation": cmsis_nn_activation
    }


class cmsis_nn_pool_params(Struct):
    typeName = "cmsis_nn_pool_params"
    structTypeDict = {"stride": cmsis_nn_tile, "padding": cmsis_nn_tile, "activation": cmsis_nn_activation}


class cmsis_nn_dw_conv_params(Struct):
    typeName = "cmsis_nn_dw_conv_params"
    structTypeDict = {
        "input_offset": int32_t,
        "output_offset": int32_t,
        "ch_mult": int32_t,
        "stride": cmsis_nn_tile,
        "padding": cmsis_nn_tile,
        "dilation": cmsis_nn_tile,
        "activation": cmsis_nn_activation
    }
