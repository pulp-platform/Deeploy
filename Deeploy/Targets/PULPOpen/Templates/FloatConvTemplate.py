# ----------------------------------------------------------------------
#
# File: FLoatConvTemplate.py
#
# Last edited: 23.01.2025
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
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


class PULP2DFloatConvIm2ColTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)

    @staticmethod
    def computeTransientBuffersSize(
            ctxt: NetworkContext,
            operatorRepresentation: OperatorRepresentation) -> List[Tuple[str, Union[int, IntVar]]]:
        im2col_dim = 4 * 8 * (operatorRepresentation['ch_im_in'] * operatorRepresentation['dim_kernel_x'] *
                              operatorRepresentation['dim_kernel_y'])
        im2col_name = operatorRepresentation['nodeName'] + "_buffer"
        return [(im2col_name, im2col_dim)]

    def hoistTransientBuffers(self, ctxt: NetworkContext,
                              operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
        im2col_name, im2col_dim = PULP2DFloatConvIm2ColTemplate.computeTransientBuffersSize(
            ctxt, operatorRepresentation)[0]
        ctxt.hoistTransientBuffer(im2col_name, im2col_dim)

        operatorRepresentation['ctxtBuffer'] = im2col_name
        operatorRepresentation['ctxtBufferSize'] = im2col_dim
        return ctxt, operatorRepresentation, [im2col_name]


reference2DTemplate = NodeTemplate("""

// 2D FP Conv HWC Parallel (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int16_t ${nodeName}_ch_out_chunk = (${ch_im_out} >> ${nodeName}_log2Core) + ((${ch_im_out} & (NUM_CORES-1))!=0);
int16_t ${nodeName}_ch_out_start = MIN(${nodeName}_ch_out_chunk*${nodeName}_core_id, ${ch_im_out});
int16_t ${nodeName}_ch_out_stop = MIN(${nodeName}_ch_out_start + ${nodeName}_ch_out_chunk, ${ch_im_out});
int16_t ${nodeName}_ch_out_count = ${nodeName}_ch_out_stop - ${nodeName}_ch_out_start;

${weight_type.typeName} ${nodeName}_weight_ptr = ${weight} + ${nodeName}_ch_out_start * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y};

${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};
                                   

for (uint32_t n=0; n<${batch}; ++n) {

    Conv2d_ChannelRange_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in}, ${dim_im_in_y}, ${dim_im_in_x}, ${ch_im_in},
        ${nodeName}_weight_ptr, ${nodeName}_ch_out_count,
        ${dim_kernel_y}, ${dim_kernel_x},
        ${stride_y}, ${stride_x},
        ref_${data_out}_${data_out}, ${ch_im_out}, ${nodeName}_ch_out_start,
        ${padding_y_top}, ${padding_y_bottom}, ${padding_x_left}, ${padding_x_right}
    );
    

    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}

""")

reference2DIm2ColTemplate = PULP2DFloatConvIm2ColTemplate("""
// 2D FP Conv HWC Parallel with Im2Col (Name: ${nodeName}, Op: ${nodeOp})
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int16_t ${nodeName}_ch_out_chunk = (${ch_im_out} >> ${nodeName}_log2Core) + ((${ch_im_out} & (NUM_CORES-1))!=0);
int16_t ${nodeName}_ch_out_start = MIN(${nodeName}_ch_out_chunk*${nodeName}_core_id, ${ch_im_out});
int16_t ${nodeName}_ch_out_stop = MIN(${nodeName}_ch_out_start + ${nodeName}_ch_out_chunk, ${ch_im_out});
int16_t ${nodeName}_ch_out_count = ${nodeName}_ch_out_stop - ${nodeName}_ch_out_start;

${weight_type.typeName} ${nodeName}_weight_ptr = ${weight} + ${nodeName}_ch_out_start * ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y};


uint32_t ${nodeName}_im2col_size_per_core = ${ch_im_in} * ${dim_kernel_x} * ${dim_kernel_y};
${data_out_type.typeName} ${nodeName}_im2col_buffer = ((${data_out_type.typeName})${ctxtBuffer}) + ${nodeName}_core_id * ${nodeName}_im2col_size_per_core;
                                                          
${data_in_type.typeName} ref_${data_out}_${data_in} = ${data_in};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for (uint32_t n=0; n<${batch}; ++n) {   
    
    Conv2d_Im2Col_ChannelRange_fp${data_in_type.referencedType.typeWidth}_fp${weight_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}_HWC(
        ref_${data_out}_${data_in},            
        ${dim_im_in_y},                      
        ${dim_im_in_x},                      
        ${ch_im_in},                          
        ${nodeName}_weight_ptr,               
        ${nodeName}_ch_out_count,            
        ${dim_kernel_y},                      
        ${dim_kernel_x},                      
        ${stride_y},                          
        ${stride_x},                          
        ref_${data_out}_${data_out},         
        ${ch_im_out},                         
        ${nodeName}_ch_out_start,             
        ${padding_y_top},                    
        ${padding_y_bottom},                  
        ${padding_x_left},                    
        ${padding_x_right},                   
        ${nodeName}_im2col_buffer             
    );

    ref_${data_out}_${data_in} += ${ch_im_in} * ${dim_im_in_x} * ${dim_im_in_y};
    ref_${data_out}_${data_out} += ${ch_im_out} * ${dim_im_out_x} * ${dim_im_out_y};
}
""")
