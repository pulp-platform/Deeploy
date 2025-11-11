# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Add Parallel with 1x6 unrolling (Name: ${nodeName}, Op: ${nodeOp})
uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});

uint32_t i = ${nodeName}_chunk_start;
for (; i + 5 < ${nodeName}_chunk_stop; i += 6) {
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