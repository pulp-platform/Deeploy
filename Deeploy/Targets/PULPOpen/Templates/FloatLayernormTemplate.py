# ----------------------------------------------------------------------
#
# File: FloatLayernormTemplate.py
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

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// FloatLayernorm Parallel (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);

int32_t ${nodeName}_seq_length = ${size} / ${lastDimLength};
int32_t ${nodeName}_chunk = (${nodeName}_seq_length >> ${nodeName}_log2Core) + 
                          ((${nodeName}_seq_length & (NUM_CORES-1)) != 0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, ${nodeName}_seq_length);
int32_t ${nodeName}_end = MIN(${nodeName}_start + ${nodeName}_chunk, ${nodeName}_seq_length);


int32_t ${nodeName}_elem_start = ${nodeName}_start * ${lastDimLength};
int32_t ${nodeName}_elem_end = ${nodeName}_end * ${lastDimLength};
int32_t ${nodeName}_elem_count = ${nodeName}_elem_end - ${nodeName}_elem_start;


const float* ${nodeName}_data_in_ptr = ${data_in} + ${nodeName}_elem_start;
float* ${nodeName}_data_out_ptr = ${data_out} + ${nodeName}_elem_start;


if (${nodeName}_elem_count > 0) {
    Layernorm_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
        ${nodeName}_data_in_ptr, 
        ${nodeName}_data_out_ptr, 
        ${weight}, 
        ${bias}, 
        ${epsilon}, 
        ${nodeName}_elem_count, 
        ${lastDimLength}
    );
}

""")