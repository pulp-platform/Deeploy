# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate


class PULPDequantTemplate(NodeTemplate):

    def __init__(self, templateStr):
        super().__init__(templateStr)


referenceTemplate = PULPDequantTemplate("""
// Dequantization (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});


for (uint32_t i=${nodeName}_chunk_start; i<${nodeName}_chunk_stop; i++) {
    int32_t quantized = (int32_t)${data_in}[i];
    float32_t shifted_val = quantized - ${zero_point};
    float32_t dequantized = shifted_val * ${scale};

    ${data_out}[i] = (${data_out_type.referencedType.typeName})dequantized;
}
""")
