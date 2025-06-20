# ----------------------------------------------------------------------
#
# File: FloatGemmTemplate.py.py
#
# Last edited: 05.06.2025
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

class PULPFloatGEMMTemplate(NodeTemplate):
    
    def __init__(self, templateStr):
        super().__init__(templateStr)
    
    def alignToContext(self, ctxt: NetworkContext,
                      operatorRepresentation: OperatorRepresentation) -> Tuple[NetworkContext, Dict, List[str]]:
    
        if 'C' not in operatorRepresentation or operatorRepresentation['C'] is None:
            # No bias case - set C to NULL and provide a default type
            operatorRepresentation['C'] = None
            operatorRepresentation['C_type'] = float32_tPtr  # Default to fp32 type
        
        return ctxt, operatorRepresentation, []

referenceTemplate = PULPFloatGEMMTemplate("""
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
${A_type.typeName} ref_${data_out}_${A} = ${A};
${B_type.typeName} ref_${data_out}_${B} = ${B};
% if C is not None:
${C_type.typeName} ref_${data_out}_${C} = ${C};
% else:
${C_type.typeName} ref_${data_out}_C = NULL;
% endif
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0; i<${batch}; i++){
    % if C is not None:
    PULP_Gemm_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${C_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        ref_${data_out}_${C},
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ${transA},
        ${transB}
    );
    % else:
    PULP_Gemm_fp${A_type.referencedType.typeWidth}_fp${B_type.referencedType.typeWidth}_fp${C_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
        ref_${data_out}_${A},
        ref_${data_out}_${B},
        NULL,
        ref_${data_out}_${data_out},
        ${M},
        ${N},
        ${O},
        ${transA},
        ${transB}
    );
    % endif
    
    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    % if C is not None:
    ref_${data_out}_${C} += ${M} * ${O};
    % endif
    ref_${data_out}_${data_out} += ${M} * ${O};
}
""")