# ----------------------------------------------------------------------
#
# File: FloatMulTemplate.py
#
# Last edited: 05.06.2025
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Authors:
# - Run Wang, ETH Zurich
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
// Float Mul with parallelism and 6x unrolling (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1)) != 0);
uint32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, ${size});
uint32_t ${nodeName}_end = MIN(${nodeName}_start + ${nodeName}_chunk, ${size});

if (${nodeName}_start < ${nodeName}_end) {
    float32_t ${nodeName}_scalar = ${B}[0];
    uint32_t ${nodeName}_unroll_end = ${nodeName}_start + ((${nodeName}_end - ${nodeName}_start) / 6) * 6;
    for (uint32_t i = ${nodeName}_start; i < ${nodeName}_unroll_end; i += 6) {
        ${C}[i + 0] = ${A}[i + 0] * ${nodeName}_scalar;
        ${C}[i + 1] = ${A}[i + 1] * ${nodeName}_scalar;
        ${C}[i + 2] = ${A}[i + 2] * ${nodeName}_scalar;
        ${C}[i + 3] = ${A}[i + 3] * ${nodeName}_scalar;
        ${C}[i + 4] = ${A}[i + 4] * ${nodeName}_scalar;
        ${C}[i + 5] = ${A}[i + 5] * ${nodeName}_scalar;
    }
    for (uint32_t i = ${nodeName}_unroll_end; i < ${nodeName}_end; i++) {
        ${C}[i] = ${A}[i] * ${nodeName}_scalar;
    }
}
""")