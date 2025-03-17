# ----------------------------------------------------------------------
#
# File: QuantTemplate.py
#
# Last edited: 12.03.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author: Federico Brancasi, ETH Zurich
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


class _QuantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _QuantTemplate("""
// Quantization (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
    
    for (uint32_t i=0; i<${size}; i++) {
        // quantization formula
        float32_t input_val = ${data_in}[i];
        float32_t scaled_val = input_val * ${scale};  // Multiply instead of divide
        float32_t shifted_val = scaled_val + ${zero_point};
        
        // Round to nearest integer
        int32_t quantized = (int32_t)(shifted_val + 0.5f * (shifted_val >= 0 ? 1 : -1));
        
        // Clamp the value
        if (quantized < ${min_val}) quantized = ${min_val};
        if (quantized > ${max_val}) quantized = ${max_val};
        
        // Assign directly with explicit cast
        ${data_out}[i] = (${data_out_type.referencedType.typeName})quantized;
        
    }
END_SINGLE_CORE
""")
