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

for(uint32_t b=0; b<${batch}; b++) {
    ${A_type.typeName} batch_A = ${A} + b * ${M} * ${N};
    ${B_type.typeName} batch_B = ${B} + b * ${N} * ${O};
    ${data_out_type.typeName} batch_out = ${data_out} + b * ${M} * ${O};
    
    PULP_MatMul_fp32_fp32_fp32_unroll1x7(
        batch_A,
        batch_B, 
        batch_out,
        ${M},
        ${N}, 
        ${O}
    );
}
""")