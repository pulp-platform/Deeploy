# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

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
