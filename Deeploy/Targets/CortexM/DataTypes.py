# ----------------------------------------------------------------------
#
# File: CMSISDataTypes.py
#
# Last edited: 01.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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
