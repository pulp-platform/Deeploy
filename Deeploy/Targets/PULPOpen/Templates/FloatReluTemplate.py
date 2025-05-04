# ----------------------------------------------------------------------
#
# File: FloatReluTemplate.py
#
# Last edited: 04.05.2025
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
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Parallel ReLU (Name: ${nodeName}, Op: ${nodeOp})
int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk*${nodeName}_core_id, ${size});
int32_t ${nodeName}_end = MIN(${nodeName}_start + ${nodeName}_chunk, ${size});
int32_t ${nodeName}_local_size = ${nodeName}_end - ${nodeName}_start;

// Call the original function with adjusted pointers and size
if (${nodeName}_local_size > 0) {
    Relu_fp${data_in_type.referencedType.typeWidth}_fp${data_out_type.referencedType.typeWidth}(
        ${data_in} + ${nodeName}_start,
        ${data_out} + ${nodeName}_start,
        ${nodeName}_local_size
    );
}
""")