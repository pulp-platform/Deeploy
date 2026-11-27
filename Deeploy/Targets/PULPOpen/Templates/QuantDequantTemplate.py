# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

_chunk = """
int32_t ${nodeName}_size = ${size};
int8_t ${nodeName}_log2Core = LOG2(NUM_CORES);
int32_t ${nodeName}_chunk = (${nodeName}_size >> ${nodeName}_log2Core) + ((${nodeName}_size & (NUM_CORES - 1)) != 0);
int32_t ${nodeName}_start = MIN(${nodeName}_chunk * (int32_t)pi_core_id(), ${nodeName}_size);
int32_t ${nodeName}_end = MIN(${nodeName}_start + ${nodeName}_chunk, ${nodeName}_size);
"""

quantTemplate = NodeTemplate("""
// Quantization (Name: ${nodeName}, Op: ${nodeOp})""" + _chunk + """
for (int32_t i = ${nodeName}_start; i < ${nodeName}_end; i++) {
    float32_t scaled_val = ${data_in}[i] * (float32_t)${scale};
    float32_t shifted_val = scaled_val + (float32_t)${zero_point};
    int32_t quantized = (int32_t)(shifted_val + 0.5f * (shifted_val >= 0 ? 1 : -1));
    if (quantized < ${min_val}) quantized = ${min_val};
    if (quantized > ${max_val}) quantized = ${max_val};
    ${data_out}[i] = (${data_out_type.referencedType.typeName})quantized;
}
""")

dequantTemplate = NodeTemplate("""
// Dequantization (Name: ${nodeName}, Op: ${nodeOp})""" + _chunk + """
for (int32_t i = ${nodeName}_start; i < ${nodeName}_end; i++) {
% if not float(zero_point).is_integer():
    float32_t shifted_val = (float32_t)${data_in}[i] - (float32_t)${zero_point};
% elif zero_point == 0 or data_in_type.referencedType.typeWidth < 32:
    float32_t shifted_val = (float32_t)((int32_t)${data_in}[i] - ${int(zero_point)});
% else:
    float32_t shifted_val = (float32_t)((int64_t)${data_in}[i] - ${int(zero_point)});
% endif
    ${data_out}[i] = (${data_out_type.referencedType.typeName})(shifted_val * (float32_t)${scale});
}
""")
