# ----------------------------------------------------------------------
#
# File: MulTemplate.py
#
# Last edited: 02.09.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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
// Mul Parallel with 1x6 unrolling (Name: ${nodeName}, Op: ${nodeOp})
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int16_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int16_t ${nodeName}_chunk_start = MIN(${nodeName}_chunk*${nodeName}_core_id, ${size});
int16_t ${nodeName}_chunk_stop = MIN(${nodeName}_chunk_start + ${nodeName}_chunk, ${size});

uint32_t i = ${nodeName}_chunk_start;
for (; i+5 < ${nodeName}_chunk_stop; i+=6) {
    ${C}[i] = ${A}[i] * ${B}[0];
    ${C}[i+1] = ${A}[i+1] * ${B}[0];
    ${C}[i+2] = ${A}[i+2] * ${B}[0];
    ${C}[i+3] = ${A}[i+3] * ${B}[0];
    ${C}[i+4] = ${A}[i+4] * ${B}[0];
    ${C}[i+5] = ${A}[i+5] * ${B}[0];
}

for (; i < ${nodeName}_chunk_stop; i++) {
    ${C}[i] = ${A}[i] * ${B}[0];
}
""")
