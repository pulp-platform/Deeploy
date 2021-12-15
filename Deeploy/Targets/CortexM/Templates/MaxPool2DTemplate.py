# ----------------------------------------------------------------------
#
# File: MaxPool2DTemplate.py
#
# Last edited: 27.12.2021
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

from typing import Dict, List, Tuple

from Deeploy.DeeployTypes import NetworkContext, NodeTemplate, OperatorRepresentation
from Deeploy.Targets.CortexM.DataTypes import cmsis_nn_context, cmsis_nn_dims, cmsis_nn_pool_params


class _MaxPool2DTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    def alignToContext(self, ctxt: NetworkContext,
                       operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:

        nameList = []

        data_out_name = operatorRepresentation['data_out']

        ctxtDict = {'buf': None, 'size': 0}

        nameList += [ctxt.hoistStruct(ctxtDict, f'{data_out_name}_ctxt', cmsis_nn_context)]
        operatorRepresentation['ctxt'] = f'{data_out_name}_ctxt'

        strideDict = {'h': operatorRepresentation['stride_x'], 'w': operatorRepresentation['stride_y']}
        paddingDict = {'h': operatorRepresentation['padding_x'], 'w': operatorRepresentation['padding_y']}
        activationDict = {'min': -2**7, 'max': 2**7 - 1}

        convParamsDict = {
            'stride': strideDict,
            'padding': paddingDict,
            'activation': activationDict,
        }
        nameList += [ctxt.hoistStruct(convParamsDict, f'{data_out_name}_pool_params', cmsis_nn_pool_params)]
        operatorRepresentation['pool_params'] = ctxt.lookup(f'{data_out_name}_pool_params').name

        inputDimsDict = {
            'n': 1,
            'h': operatorRepresentation['dim_im_in_x'],
            'w': operatorRepresentation['dim_im_in_y'],
            'c': operatorRepresentation['ch_im_in']
        }
        nameList += [ctxt.hoistStruct(inputDimsDict, f'{data_out_name}_input_dims', cmsis_nn_dims)]
        operatorRepresentation['input_dims'] = ctxt.lookup(f'{data_out_name}_input_dims').name

        filterDimsDict = {
            'n': 1,
            'h': operatorRepresentation['dim_kernel_x'],
            'w': operatorRepresentation['dim_kernel_y'],
            'c': 1
        }
        nameList += [ctxt.hoistStruct(filterDimsDict, f'{data_out_name}_filter_dims', cmsis_nn_dims)]
        operatorRepresentation['filter_dims'] = ctxt.lookup(f'{data_out_name}_filter_dims').name

        outputDimsDict = {
            'n': 1,
            'h': operatorRepresentation['dim_im_out_x'],
            'w': operatorRepresentation['dim_im_out_y'],
            'c': operatorRepresentation['ch_im_out']
        }
        nameList += [ctxt.hoistStruct(outputDimsDict, f'{data_out_name}_output_dims', cmsis_nn_dims)]
        operatorRepresentation['output_dims'] = ctxt.lookup(f'{data_out_name}_output_dims').name

        return ctxt, operatorRepresentation, nameList


cmsisTemplate = _MaxPool2DTemplate("""
<%
batchSizeIn = dim_im_in_x * dim_im_in_y * ch_im_in
batchSizeOut = dim_im_out_x * dim_im_out_y * ch_im_out
%>
// MaxPool2D
for(int b=0;b<${batch};b++){
arm_max_pool_s8(&${ctxt}, &${pool_params}, &${input_dims}, (${data_in}+b*${batchSizeIn}), &${filter_dims}, &${output_dims}, (${data_out} + b*${batchSizeOut}));
}
""")
