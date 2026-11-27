# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class _DequantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = _DequantTemplate("""
// Dequantization (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE

    for (uint32_t i=0; i<${size}; i++) {
% if not float(zero_point).is_integer():
        float32_t shifted_val = (float32_t)${data_in}[i] - (float32_t)${zero_point};
% elif zero_point == 0 or data_in_type.referencedType.typeWidth < 32:
        float32_t shifted_val = (float32_t)((int32_t)${data_in}[i] - ${int(zero_point)});
% else:
        float32_t shifted_val = (float32_t)((int64_t)${data_in}[i] - ${int(zero_point)});
% endif
        float32_t dequantized = shifted_val * (float32_t)${scale};

        ${data_out}[i] = (${data_out_type.referencedType.typeName})dequantized;
    }

END_SINGLE_CORE
""")
