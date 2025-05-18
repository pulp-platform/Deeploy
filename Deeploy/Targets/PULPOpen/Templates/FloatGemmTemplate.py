# ----------------------------------------------------------------------
#
# File: GemmTemplate.py.py
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
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int32_t ${nodeName}_M_chunk = (${M} >> ${nodeName}_log2Core) + ((${M} & (NUM_CORES-1))!=0);
int32_t ${nodeName}_M_start = MIN(${nodeName}_core_id * ${nodeName}_M_chunk, ${M});
int32_t ${nodeName}_M_end = MIN(${nodeName}_M_start + ${nodeName}_M_chunk, ${M});
int32_t ${nodeName}_M_size = ${nodeName}_M_end - ${nodeName}_M_start;

${A_type.typeName} ref_${data_out}_${A} = ${A};
${B_type.typeName} ref_${data_out}_${B} = ${B};
${C_type.typeName} ref_${data_out}_${C} = ${C};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0; i<${batch}; i++) {
    ${A_type.typeName} batch_A = ref_${data_out}_${A} + i * ${M} * ${N};
    ${B_type.typeName} batch_B = ref_${data_out}_${B} + i * ${N} * ${O};
    ${C_type.typeName} batch_C = ref_${data_out}_${C} + i * ${M} * ${O};
    ${data_out_type.typeName} batch_out = ref_${data_out}_${data_out} + i * ${M} * ${O};
    
    if (${nodeName}_M_size > 0) {
        Gemm_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${C_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
            batch_A + ${nodeName}_M_start * ${N},
            batch_B,
            batch_C + ${nodeName}_M_start * ${O},
            batch_out + ${nodeName}_M_start * ${O},
            ${nodeName}_M_size,
            ${N},
            ${O},
            ${transA},
            ${transB}
        );
    }
}
""")