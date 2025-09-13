# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

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
