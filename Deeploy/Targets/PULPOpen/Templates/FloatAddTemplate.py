# ----------------------------------------------------------------------
#
# File: FloatAddTemplate.py
#
# Last edited: 13.11.2024
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
// Add Parallel with 1x6 unrolling (Name: ${nodeName}, Op: ${nodeOp})
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int16_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int16_t ${nodeName}_chunk_start = MIN(${nodeName}_chunk*${nodeName}_core_id, ${size});
int16_t ${nodeName}_chunk_stop = MIN(${nodeName}_chunk_start + ${nodeName}_chunk, ${size});

uint32_t i = ${nodeName}_chunk_start;
for (; i+5 < ${nodeName}_chunk_stop; i+=6) {
    ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[i];
    ${data_out}[i+1] = ${data_in_1}[i+1] + ${data_in_2}[i+1];
    ${data_out}[i+2] = ${data_in_1}[i+2] + ${data_in_2}[i+2];
    ${data_out}[i+3] = ${data_in_1}[i+3] + ${data_in_2}[i+3];
    ${data_out}[i+4] = ${data_in_1}[i+4] + ${data_in_2}[i+4];
    ${data_out}[i+5] = ${data_in_1}[i+5] + ${data_in_2}[i+5];
}

for (; i < ${nodeName}_chunk_stop; i++) {
    ${data_out}[i] = ${data_in_1}[i] + ${data_in_2}[i];
}
""")