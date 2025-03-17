# ----------------------------------------------------------------------

# File: DequantTemplate.py

# Last edited: 17.03.2025

# Copyright (C) 2025, ETH Zurich and University of Bologna.

# Author: Federico Brancasi, ETH Zurich

# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at

# www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from Deeploy.DeeployTypes import NodeTemplate


class _DequantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _DequantTemplate("""
// Dequantization (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

    for (uint32_t i=0; i<${size}; i++) {
        int32_t quantized = (int32_t)${data_in}[i];
        float32_t shifted_val = quantized - ${zero_point};
        float32_t dequantized = shifted_val * ${scale};

        ${data_out}[i] = (${data_out_type.referencedType.typeName})dequantized;
    }

END_SINGLE_CORE
""")
