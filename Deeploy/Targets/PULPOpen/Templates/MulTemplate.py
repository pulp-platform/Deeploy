# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate, OperatorRepresentation


class _MulTemplate(NodeTemplate, OperatorRepresentation):
    pass


# Element-wise, or with a single-element B (sizeB == 1, see MulParser). The chunk bounds are clamped to size so that
# no core touches index size or beyond.
referenceTemplate = _MulTemplate("""
// Mul (Name: ${nodeName}, Op: ${nodeOp})

uint32_t ${nodeName}_core_id = pi_core_id();
uint32_t ${nodeName}_log2Core = (uint32_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});

#pragma unroll 2
for (uint32_t i=${nodeName}_chunk_start;i<${nodeName}_chunk_stop;i++){
    ${C}[i] = ${A}[i] * ${B}[${0 if sizeB == 1 else 'i'}];
}

""")
