# ----------------------------------------------------------------------
#
# File: FloatSoftmaxTemplate.py
#
# Last edited: 23.1.2025
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
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
// Softmax with external function call (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_num_cores = NUM_CORES;

int32_t ${nodeName}_num_vectors = ${size} / ${lastDimLength};

int32_t ${nodeName}_vectors_per_core = (${nodeName}_num_vectors + ${nodeName}_num_cores - 1) / ${nodeName}_num_cores;
int32_t ${nodeName}_vector_start = MIN(${nodeName}_core_id * ${nodeName}_vectors_per_core, ${nodeName}_num_vectors);
int32_t ${nodeName}_vector_end = MIN(${nodeName}_vector_start + ${nodeName}_vectors_per_core, ${nodeName}_num_vectors);


int32_t ${nodeName}_local_size = (${nodeName}_vector_end - ${nodeName}_vector_start) * ${lastDimLength};

if (${nodeName}_local_size > 0) {

    int32_t ${nodeName}_data_offset = ${nodeName}_vector_start * ${lastDimLength};
    
    Softmax_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
        ${data_in} + ${nodeName}_data_offset,
        ${data_out} + ${nodeName}_data_offset,
        ${nodeName}_local_size,
        ${lastDimLength}
    );
}

""")

referenceGradientTemplate = NodeTemplate("""
// Softmax Gradient (Name: ${nodeName}, Op: ${nodeOp})
SINGLE_CORE SoftmaxGrad_fp32_fp32_fp32(${upstream_grad}, ${softmax_output}, ${softmax_grad}, ${size}, ${lastDimLength});
""")