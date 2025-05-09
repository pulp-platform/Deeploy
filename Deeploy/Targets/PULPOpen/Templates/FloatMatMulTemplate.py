# ----------------------------------------------------------------------
#
# File: F；FloatMatMul.py
#
# Last edited: 28.03.2025
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
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int32_t ${nodeName}_M_chunk = (${M} >> ${nodeName}_log2Core) + ((${M} & (NUM_CORES-1))!=0);
int32_t ${nodeName}_M_start = MIN(${nodeName}_core_id * ${nodeName}_M_chunk, ${M});
int32_t ${nodeName}_M_end = MIN(${nodeName}_M_start + ${nodeName}_M_chunk, ${M});
int32_t ${nodeName}_M_size = ${nodeName}_M_end - ${nodeName}_M_start;
                                 
for(uint32_t b=0; b<${batch}; b++) {
    ${A_type.typeName} batch_A = ${A} + b * ${M} * ${N};
    ${B_type.typeName} batch_B = ${B} + b * ${N} * ${O};
    ${data_out_type.typeName} batch_out = ${data_out} + b * ${M} * ${O};
    
    if (${nodeName}_M_size > 0) {
        MatMul_fp32_fp32_fp32_unroll1x7(
            batch_A + ${nodeName}_M_start * ${N},  
            batch_B,                              
            batch_out + ${nodeName}_M_start * ${O}, 
            ${nodeName}_M_size,                    
            ${N},                                  
            ${O}                                  
        );
    }
}
""")