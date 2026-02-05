# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate, OperatorRepresentation


class _MulTemplate(NodeTemplate, OperatorRepresentation):
    pass


referenceTemplate = _MulTemplate("""
// Mul (Name: ${nodeName}, Op: ${nodeOp})

int8_t ${nodeName}_core_id = pi_core_id();
int8_t ${nodeName}_log2Core = log2(NUM_CORES);
int16_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
int16_t ${nodeName}_chunk_start = MIN(${nodeName}_chunk*${nodeName}_core_id, ${size});
int16_t ${nodeName}_chunk_stop = MIN(${nodeName}_chunk_start + ${nodeName}_chunk, ${size} + 1);

#pragma unroll 2
for (uint32_t i=${nodeName}_chunk_start;i<${nodeName}_chunk_stop;i++){
    ${C}[i] = ${A}[i] * ${B}[i];
}

""")
