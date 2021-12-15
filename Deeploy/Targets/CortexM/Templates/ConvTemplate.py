# ----------------------------------------------------------------------
#
# File: ConvTemplate.py
#
# Last edited: 17.12.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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

from typing import Dict, List, Tuple, Union

from ortools.constraint_solver.pywrapcp import IntVar

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.CortexM.DataTypes import cmsis_nn_context, cmsis_nn_conv_params, cmsis_nn_dims, \
    cmsis_nn_per_channel_quant_params


class _Conv2D_8_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        nameList = []

        data_out_name = operatorRepresentation['data_out']
        ctxtDict = {'buf': operatorRepresentation['ctxtBuffer'], 'size': operatorRepresentation['ctxtBufferSize']}
        nameList += [ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', cmsis_nn_context)]
        operatorRepresentation['ctxt'] = f'{data_out_name}_ctxt'

        strideDict = {
            'h': operatorRepresentation['stride_x'],
            'w': operatorRepresentation['stride_y'],
        }
        paddingDict = {'h': operatorRepresentation['padding_x'], 'w': operatorRepresentation['padding_y']}
        dilationDict = {'h': operatorRepresentation['dilation_x'], 'w': operatorRepresentation['dilation_y']}
        activationDict = {
            'min': -(operatorRepresentation['n_levels'] // 2),
            'max': (operatorRepresentation['n_levels'] // 2) - 1
        }

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        assert data_in._signed is not None
        assert data_out._signed is not None

        convParamsDict = cmsis_nn_conv_params({
            'input_offset': (data_in._signed == 0) * operatorRepresentation['n_levels'] // 2,
            'output_offset': -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2,
            'stride': strideDict,
            'padding': paddingDict,
            'dilation': dilationDict,
            'activation': activationDict,
        })
        nameList += [ctxt.hoistStruct(convParamsDict, f'{data_out_name}_conv_params', cmsis_nn_conv_params)]
        operatorRepresentation['conv_params'] = ctxt.lookup(f'{data_out_name}_conv_params').name

        convQuantDict = cmsis_nn_per_channel_quant_params(
            {
                'multiplier': operatorRepresentation['mul'],
                'shift': operatorRepresentation['shift'],
            }, ctxt)

        nameList += [
            ctxt.hoistStruct(convQuantDict, f'{data_out_name}_quant_params', cmsis_nn_per_channel_quant_params)
        ]
        operatorRepresentation['quant_params'] = ctxt.lookup(f'{data_out_name}_quant_params').name

        inputDimsDict = cmsis_nn_dims({
            'n': data_in.shape[0],
            'h': operatorRepresentation['dim_im_in_x'],
            'w': operatorRepresentation['dim_im_in_y'],
            'c': operatorRepresentation['ch_im_in']
        })
        nameList += [ctxt.hoistStruct(inputDimsDict, f'{data_out_name}_input_dims', cmsis_nn_dims)]
        operatorRepresentation['input_dims'] = ctxt.lookup(f'{data_out_name}_input_dims').name

        filterDimsDict = cmsis_nn_dims({
            'n': operatorRepresentation['ch_im_out'],
            'h': operatorRepresentation['dim_kernel_x'],
            'w': operatorRepresentation['dim_kernel_y'],
            'c': operatorRepresentation['ch_im_in']
        })
        nameList += [ctxt.hoistStruct(filterDimsDict, f'{data_out_name}_filter_dims', cmsis_nn_dims)]
        operatorRepresentation['filter_dims'] = ctxt.lookup(f'{data_out_name}_filter_dims').name

        outputDimsDict = cmsis_nn_dims({
            'n': data_in.shape[0],
            'h': operatorRepresentation['dim_im_out_x'],
            'w': operatorRepresentation['dim_im_out_y'],
            'c': operatorRepresentation['ch_im_out']
        })
        nameList += [ctxt.hoistStruct(outputDimsDict, f'{data_out_name}_output_dims', cmsis_nn_dims)]
        operatorRepresentation['output_dims'] = ctxt.lookup(f'{data_out_name}_output_dims').name

        biasDimsDict = cmsis_nn_dims({
            'n': 1,
            'h': 1,
            'w': 1,
            'c': operatorRepresentation['ch_im_out'],
        })
        nameList += [ctxt.hoistStruct(biasDimsDict, f'{data_out_name}_bias_dims', cmsis_nn_dims)]
        operatorRepresentation['bias_dims'] = ctxt.lookup(f'{data_out_name}_bias_dims').name

        return ctxt, operatorRepresentation, nameList

    def computeTransientBuffersSize(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        size = 2 * operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] * operatorRepresentation[
            'dim_kernel_y'] * 2
        name = operatorRepresentation['nodeName'] + f"_buffer"
        return [(name, size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # SCHEREMO: Hoist transient buffer
        # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c

        name, size = self.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(name, size)
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, [name]


cmsis2D_8_Template = _Conv2D_8_Template("\
arm_convolve_wrapper_s8(&${ctxt}, &${conv_params}, &${quant_params}, &${input_dims}, ${data_in}, &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, ${data_out}); \n\
")


class _Conv1D_16_Template(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        data_out_name = operatorRepresentation['data_out']
        nameList = []

        # Hoist the structs to the global ctxt
        ctxtDict = {'buf': operatorRepresentation['ctxtBuffer'], 'size': operatorRepresentation['ctxtBufferSize']}
        nameList += [ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', cmsis_nn_context)]
        operatorRepresentation['ctxt'] = f'{data_out_name}_ctxt'

        # Next the conv params
        # stride
        strideDict = {
            'h': 1,
            'w': operatorRepresentation['stride_y'],
        }
        paddingDict = {'h': 0, 'w': operatorRepresentation['padding_y']}
        dilationDict = {'h': 1, 'w': operatorRepresentation['dilation_y']}
        activationDict = {
            'min': -(operatorRepresentation['n_levels'] // 2),
            'max': (operatorRepresentation['n_levels'] // 2) - 1
        }

        data_in = ctxt.lookup(operatorRepresentation['data_in'])
        data_out = ctxt.lookup(operatorRepresentation['data_out'])

        assert data_in._signed is not None
        assert data_out._signed is not None

        convParamsDict = {
            'input_offset': (data_in._signed == 0) * operatorRepresentation['n_levels'] // 2,
            'output_offset': -(data_out._signed == 0) * operatorRepresentation['n_levels'] // 2,
            'stride': strideDict,
            'padding': paddingDict,
            'dilation': dilationDict,
            'activation': activationDict,
        }
        nameList += [ctxt.hoistStruct(convParamsDict, f'{data_out_name}_conv_params', cmsis_nn_conv_params)]
        operatorRepresentation['conv_params'] = ctxt.lookup(f'{data_out_name}_conv_params').name

        convQuantDict = {
            'multiplier': operatorRepresentation['mul'],
            'shift': operatorRepresentation['shift'],
        }
        nameList += [
            ctxt.hoistStruct(convQuantDict, f'{data_out_name}_quant_params', cmsis_nn_per_channel_quant_params)
        ]
        operatorRepresentation['quant_params'] = ctxt.lookup(f'{data_out_name}_quant_params').name

        inputDimsDict = {
            'n': data_in.shape[0],
            'h': 1,
            'w': operatorRepresentation['dim_im_in_y'],
            'c': operatorRepresentation['ch_im_in']
        }
        nameList += [ctxt.hoistStruct(inputDimsDict, f'{data_out_name}_input_dims', cmsis_nn_dims)]
        operatorRepresentation['input_dims'] = ctxt.lookup(f'{data_out_name}_input_dims').name

        filterDimsDict = {
            'n': operatorRepresentation['ch_im_out'],
            'h': 1,
            'w': operatorRepresentation['dim_kernel_y'],
            'c': operatorRepresentation['ch_im_in']
        }
        nameList += [ctxt.hoistStruct(filterDimsDict, f'{data_out_name}_filter_dims', cmsis_nn_dims)]
        operatorRepresentation['filter_dims'] = ctxt.lookup(f'{data_out_name}_filter_dims').name

        outputDimsDict = {
            'n': data_in.shape[0],
            'h': 1,
            'w': operatorRepresentation['dim_im_out_y'],
            'c': operatorRepresentation['ch_im_out']
        }
        nameList += [ctxt.hoistStruct(outputDimsDict, f'{data_out_name}_output_dims', cmsis_nn_dims)]
        operatorRepresentation['output_dims'] = ctxt.lookup(f'{data_out_name}_output_dims').name

        biasDimsDict = {
            'n': 1,
            'h': 1,
            'w': 1,
            'c': operatorRepresentation['ch_im_out'],
        }
        nameList += [ctxt.hoistStruct(biasDimsDict, f'{data_out_name}_bias_dims', cmsis_nn_dims)]
        operatorRepresentation['bias_dims'] = ctxt.lookup(f'{data_out_name}_bias_dims').name

        return ctxt, operatorRepresentation, nameList

    def computeTransientBuffersSize(
            self, ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        size = 2 * operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_y'] * 2
        name = operatorRepresentation['nodeName'] + f"_buffer"
        return [(name, size)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        # SCHEREMO: Hoist transient buffer
        # https://review.trustedfirmware.org/plugins/gitiles/mirror/ARM-software/CMSIS_5/+/refs/heads/bias_for_conv/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_s8.c

        nameList = []
        name, size = self.computeTransientBuffersSize(ctxt, operatorRepresentation)[0]
        nameList += [ctxt.hoistTransientBuffer(name, size)]
        operatorRepresentation['ctxtBuffer'] = name
        operatorRepresentation['ctxtBufferSize'] = size
        return ctxt, operatorRepresentation, nameList


cmsis1D_16_Template = _Conv1D_16_Template("""
arm_convolve_wrapper_s16(&${ctxt}, &${conv_params}, &${quant_params}, &${input_dims}, ${data_in}, &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, ${data_out});
""")

cmsis1D_8_Template = _Conv1D_16_Template("""
arm_convolve_wrapper_s8(&${ctxt}, &${conv_params}, &${quant_params}, &${input_dims}, ${data_in}, &${filter_dims}, ${weight}, &${bias_dims}, ${add}, &${output_dims}, ${data_out});
""")
