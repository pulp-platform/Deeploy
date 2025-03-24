# ----------------------------------------------------------------------
#
# File: MatMul.py.py
#
# Last edited: 27.01.2025
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
# limitations under the Licens
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Matmul with row parallelism (Name: ${nodeName}, Op: ${nodeOp})


int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_num_cores = NUM_CORES;


int32_t ${nodeName}_M_per_core = (${M} + ${nodeName}_num_cores - 1) / ${nodeName}_num_cores;
int32_t ${nodeName}_M_start = MIN(${nodeName}_core_id * ${nodeName}_M_per_core, ${M});
int32_t ${nodeName}_M_end = MIN(${nodeName}_M_start + ${nodeName}_M_per_core, ${M});


for(uint32_t b=0; b<${batch}; b++) {
 
    ${A_type.typeName} batch_A = ${A} + b * ${M} * ${N};
    ${B_type.typeName} batch_B = ${B} + b * ${N} * ${O};
    ${data_out_type.typeName} batch_out = ${data_out} + b * ${M} * ${O};
    

    for (uint32_t i = ${nodeName}_M_start; i < ${nodeName}_M_end; i++) {
        for (uint32_t j = 0; j < ${O}; j++) {
            float32_t sum = 0.0f;
            
            #pragma unroll 4
            for (uint32_t k = 0; k < ${N}; k++) {
                sum += batch_A[i * ${N} + k] * batch_B[k * ${O} + j];
            }
            
            batch_out[i * ${O} + j] = sum;
        }
    }
}

""")