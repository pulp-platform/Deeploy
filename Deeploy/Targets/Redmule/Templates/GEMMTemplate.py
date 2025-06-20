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
from Deeploy.DeeployTypes import NodeTemplate, NetworkContext, OperatorRepresentation
from Deeploy.AbstractDataTypes import float32_tPtr
from typing import Tuple, Dict, List

class RedMuleGEMMTemplate(NodeTemplate):
    
    def __init__(self, templateStr):
        super().__init__(templateStr)
    
    def alignToContext(self, ctxt: NetworkContext,
                      operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
    
        if 'C' not in operatorRepresentation or operatorRepresentation['C'] is None:
            # No bias case - set C to NULL and provide a default type
            operatorRepresentation['C'] = None
            operatorRepresentation['C_type'] = float32_tPtr  # Default to fp32 type
        
        return ctxt, operatorRepresentation, []

referenceTemplate = RedMuleGEMMTemplate("""
// GEMM using RedMule hardware accelerator (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();

if (${nodeName}_core_id == 0) {
    for(uint32_t b=0; b<${batch}; b++) {
        ${A_type.typeName} batch_A = ${A} + b * ${M} * ${N};
        ${B_type.typeName} batch_B = ${B} + b * ${N} * ${O};
        % if C is not None:
        ${C_type.typeName} batch_C = ${C} + b * ${M} * ${O};
        % endif
        ${data_out_type.typeName} batch_out = ${data_out} + b * ${M} * ${O};
        
        % if C is None or beta == 0:
        // No bias or beta=0: use MatMul
        MatMul_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_Redmule(
            (const float32_t *) batch_A,
            (const float32_t *) batch_B,
            (float32_t *) batch_out,
            ${M},
            ${N},
            ${O}
        );
        % else:
        // With bias and beta!=0: use Gemm
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
""")