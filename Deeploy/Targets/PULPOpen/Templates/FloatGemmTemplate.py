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
from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// GEMM (Name: ${nodeName}, Op: ${nodeOp})
${A_type.typeName} ref_${data_out}_${A} = ${A};
${B_type.typeName} ref_${data_out}_${B} = ${B};
${C_type.typeName} ref_${data_out}_${C} = ${C};
${data_out_type.typeName} ref_${data_out}_${data_out} = ${data_out};

for(uint32_t i=0; i<${batch}; i++){
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
    
    ref_${data_out}_${A} += ${M} * ${N};
    ref_${data_out}_${B} += ${N} * ${O};
    ref_${data_out}_${C} += ${M} * ${O};
    ref_${data_out}_${data_out} += ${M} * ${O};
}
""")