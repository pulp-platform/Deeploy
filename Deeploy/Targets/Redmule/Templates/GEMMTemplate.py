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
// GEMM using RedMule hardware accelerator (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();

if (${nodeName}_core_id == 0) {
    for(uint32_t b=0; b<${batch}; b++) {
        ${A_type.typeName} batch_A = ${A} + b * ${M} * ${N};
        ${B_type.typeName} batch_B = ${B} + b * ${N} * ${O};
        ${C_type.typeName} batch_C = ${C} + b * ${M} * ${O};
        ${data_out_type.typeName} batch_out = ${data_out} + b * ${M} * ${O};
        
        % if beta == 0:
        MatMul_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_Redmule(
            (const float32_t *) batch_A,
            (const float32_t *) batch_B,
            (float32_t *) batch_out,
            ${M},
            ${N},
            ${O}
        );
        % else:
        Gemm_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_Redmule(
            (const float32_t *) batch_A,
            (const float32_t *) batch_B,
            (const float32_t *) batch_C,
            (float32_t *) batch_out,
            ${M},
            ${N},
            ${O}
        );
        % endif
    }
}
"""
)