# ----------------------------------------------------------------------
#
# File: CMSISUtils.py
#
# Last edited: 10.01.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
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

import numpy as np

from Deeploy.Targets.CortexM.DataTypes import cmsis_nn_activation, cmsis_nn_context, cmsis_nn_conv_params, \
    cmsis_nn_dims, cmsis_nn_fc_params, cmsis_nn_per_channel_quant_params, cmsis_nn_per_tensor_quant_params


def bindConvParams(ctxt, name, repName, batch, operatorRepresentation):

    nameList = []

    # Hoist the structs to the global ctxt

    # First the context
    # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c
    bufferSize = 2 * operatorRepresentation['ch_im_in'] * operatorRepresentation[
        'dim_kernel_x'] * operatorRepresentation['dim_kernel_y'] * 2

    ctxtDict = {
        'buf': operatorRepresentation['ctxtBuffer'],  #f'{name}_ctxt_buffer',
        'size': bufferSize
    }

    nameList += [ctxt.hoistStruct(ctxtDict, f'{name}_ctxt', cmsis_nn_context)]
    operatorRepresentation[f'{repName}_ctxt'] = f'{name}_ctxt'

    # Next the conv params
    # stride
    strideDict = {
        'h': operatorRepresentation['stride_x'],
        'w': operatorRepresentation['stride_y'],
    }
    # padding
    paddingDict = {'h': operatorRepresentation['padding_x'], 'w': operatorRepresentation['padding_y']}
    # dilation
    dilationDict = {'h': operatorRepresentation['dilation_x'], 'w': operatorRepresentation['dilation_y']}
    activationDict = {
        'min': -(operatorRepresentation['n_levels'] // 2),
        'max': (operatorRepresentation['n_levels'] // 2) - 1
    }

    if 'data_in' in operatorRepresentation:
        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_in_signed = data_in._signed
    else:
        data_in_signed = operatorRepresentation['data_in_signed']

    if 'data_out' in operatorRepresentation:
        data_out = ctxt.lookup(operatorRepresentation['data_out'])
        data_out_signed = data_out._signed
    else:
        data_out_signed = operatorRepresentation['data_out_signed']

    assert data_in_signed is not None
    assert data_out_signed is not None

    convParamsDict = {
        'input_offset': (data_in_signed == 0) * operatorRepresentation['n_levels'] // 2,
        'output_offset': -(data_out_signed == 0) * operatorRepresentation['n_levels'] // 2,
        'stride': strideDict,
        'padding': paddingDict,
        'dilation': dilationDict,
        'activation': activationDict,
    }
    nameList += [ctxt.hoistStruct(convParamsDict, f'{name}_conv_params', cmsis_nn_conv_params)]
    operatorRepresentation[f'{repName}_conv_params'] = ctxt.lookup(f'{name}_conv_params').name

    convQuantDict = {
        'multiplier': ctxt._mangle(operatorRepresentation['mul']),
        'shift': ctxt._mangle(operatorRepresentation['shift']),
    }
    nameList += [ctxt.hoistStruct(convQuantDict, f'{name}_quant_params', cmsis_nn_per_channel_quant_params)]
    operatorRepresentation[f'{repName}_quant_params'] = ctxt.lookup(f'{name}_quant_params').name

    inputDimsDict = {
        'n': batch,
        'h': operatorRepresentation['dim_im_in_x'],
        'w': operatorRepresentation['dim_im_in_y'],
        'c': operatorRepresentation['ch_im_in']
    }
    nameList += [ctxt.hoistStruct(inputDimsDict, f'{name}_input_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{repName}_input_dims'] = ctxt.lookup(f'{name}_input_dims').name

    filterDimsDict = {
        'n': operatorRepresentation['ch_im_out'],
        'h': operatorRepresentation['dim_kernel_x'],
        'w': operatorRepresentation['dim_kernel_y'],
        'c': operatorRepresentation['ch_im_in']
    }
    nameList += [ctxt.hoistStruct(filterDimsDict, f'{name}_filter_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{repName}_filter_dims'] = ctxt.lookup(f'{name}_filter_dims').name

    outputDimsDict = {
        'n': batch,
        'h': operatorRepresentation['dim_im_out_x'],
        'w': operatorRepresentation['dim_im_out_y'],
        'c': operatorRepresentation['ch_im_out']
    }
    nameList += [ctxt.hoistStruct(outputDimsDict, f'{name}_output_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{repName}_output_dims'] = ctxt.lookup(f'{name}_output_dims').name

    biasDimsDict = {
        'n': 1,
        'h': 1,
        'w': 1,
        'c': operatorRepresentation['ch_im_out'],
    }
    nameList += [ctxt.hoistStruct(biasDimsDict, f'{name}_bias_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{repName}_bias_dims'] = ctxt.lookup(f'{name}_bias_dims').name

    return ctxt, operatorRepresentation


def bindFCParams(ctxt,
                 name,
                 mul,
                 shift,
                 data_in,
                 weight,
                 operatorRepresentation,
                 operatorRepresentationPrefix = '',
                 bias = True):

    nameList = []

    operatorRepresentation['in_N'] = operatorRepresentation['M']
    operatorRepresentation['in_C'] = operatorRepresentation['N']
    operatorRepresentation['weight_N'] = operatorRepresentation['N']
    operatorRepresentation['weight_C'] = operatorRepresentation['O']

    ctxtDict = {'buf': None, 'size': 0}

    nameList += [ctxt.hoistStruct(ctxtDict, f'{name}_ctxt', cmsis_nn_context)]
    operatorRepresentation[f'{operatorRepresentationPrefix}ctxt'] = f'{name}_ctxt'

    # activation
    activationDict = {
        'min': -(operatorRepresentation['n_levels'] // 2),
        'max': (operatorRepresentation['n_levels'] // 2) - 1
    }
    nameList += [ctxt.hoistStruct(activationDict, f'{name}_activation', cmsis_nn_activation)]

    data_out = ctxt.lookup(operatorRepresentation['data_out'])

    # SCHEREMO: Workaround for MHSA:
    if not hasattr(data_in, '_signed') or not hasattr(data_out, '_signed'):

        fcParamsDict = {
            'input_offset': 0,
            'output_offset': 0,
            'filter_offset': 0,
            'activation': activationDict,
        }

    else:

        fcParamsDict = {
            'input_offset': (data_in._signed == 0) * operatorRepresentation['n_levels'] // 2,
            'output_offset': -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2,
            'filter_offset': 0,
            'activation': activationDict,
        }

    nameList += [ctxt.hoistStruct(fcParamsDict, f'{name}_fc_params', cmsis_nn_fc_params)]
    operatorRepresentation[f'{operatorRepresentationPrefix}fc_params'] = ctxt.lookup(f'{name}_fc_params').name

    if isinstance(mul, str):
        __mul = ctxt.lookup(mul).values
        assert np.ndim(__mul) == 0, "Mul is not scalar!"
        _mul = __mul.item()
        ctxt.lookup(mul)._deploy = False
    else:
        _mul = mul

    if isinstance(shift, str):
        __shift = ctxt.lookup(shift).values
        assert np.ndim(__shift) == 0, "Shift is not scalar!"
        _shift = __shift.item()
        ctxt.lookup(shift)._deploy = False
    else:
        _shift = shift

    gemmQuantDict = {'multiplier': _mul, 'shift': _shift}

    nameList += [ctxt.hoistStruct(gemmQuantDict, f'{name}_quant_params', cmsis_nn_per_tensor_quant_params)]
    operatorRepresentation[f'{operatorRepresentationPrefix}quant_params'] = ctxt.lookup(f'{name}_quant_params').name

    inputDimsDict = {
        'n': operatorRepresentation['in_N'],
        'h': 1,
        'w': 1,
        'c': operatorRepresentation['in_C'],
    }
    nameList += [ctxt.hoistStruct(inputDimsDict, f'{name}_input_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{operatorRepresentationPrefix}input_dims'] = ctxt.lookup(f'{name}_input_dims').name

    filterDimsDict = {'n': operatorRepresentation['weight_N'], 'h': 1, 'w': 1, 'c': operatorRepresentation['weight_C']}
    nameList += [ctxt.hoistStruct(filterDimsDict, f'{name}_filter_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{operatorRepresentationPrefix}filter_dims'] = ctxt.lookup(f'{name}_filter_dims').name

    outputDimsDict = {'n': operatorRepresentation['in_N'], 'h': 1, 'w': 1, 'c': operatorRepresentation['weight_C']}
    nameList += [ctxt.hoistStruct(outputDimsDict, f'{name}_output_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{operatorRepresentationPrefix}output_dims'] = ctxt.lookup(f'{name}_output_dims').name

    biasDimsDict = {
        'n': 1,
        'h': 1,
        'w': 1,
        'c': operatorRepresentation['weight_C'] * bias,
    }
    nameList += [ctxt.hoistStruct(biasDimsDict, f'{name}_bias_dims', cmsis_nn_dims)]
    operatorRepresentation[f'{operatorRepresentationPrefix}bias_dims'] = ctxt.lookup(f'{name}_bias_dims').name

    return ctxt, operatorRepresentation, nameList
