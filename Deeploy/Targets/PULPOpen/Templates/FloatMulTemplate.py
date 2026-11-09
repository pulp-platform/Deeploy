# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Float Mul with parallelism and 6x unrolling (Name: ${nodeName}, Op: ${nodeOp})

uint32_t ${nodeName}_core_id = pi_core_id();
uint32_t ${nodeName}_log2Core = (uint32_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1)) != 0);
uint32_t ${nodeName}_start = MIN(${nodeName}_chunk * ${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_end = MIN(${nodeName}_start + ${nodeName}_chunk, (uint32_t) ${size});

if (${nodeName}_start < ${nodeName}_end) {
% if sizeB == 1:
    float32_t ${nodeName}_scalar = ${B}[0];
% endif
    uint32_t ${nodeName}_unroll_end = ${nodeName}_start + ((${nodeName}_end - ${nodeName}_start) / 6) * 6;
    for (uint32_t i = ${nodeName}_start; i < ${nodeName}_unroll_end; i += 6) {
% if sizeB == 1:
        ${C}[i + 0] = ${A}[i + 0] * ${nodeName}_scalar;
        ${C}[i + 1] = ${A}[i + 1] * ${nodeName}_scalar;
        ${C}[i + 2] = ${A}[i + 2] * ${nodeName}_scalar;
        ${C}[i + 3] = ${A}[i + 3] * ${nodeName}_scalar;
        ${C}[i + 4] = ${A}[i + 4] * ${nodeName}_scalar;
        ${C}[i + 5] = ${A}[i + 5] * ${nodeName}_scalar;
% else:
        ${C}[i + 0] = ${A}[i + 0] * ${B}[i + 0];
        ${C}[i + 1] = ${A}[i + 1] * ${B}[i + 1];
        ${C}[i + 2] = ${A}[i + 2] * ${B}[i + 2];
        ${C}[i + 3] = ${A}[i + 3] * ${B}[i + 3];
        ${C}[i + 4] = ${A}[i + 4] * ${B}[i + 4];
        ${C}[i + 5] = ${A}[i + 5] * ${B}[i + 5];
% endif
    }
    for (uint32_t i = ${nodeName}_unroll_end; i < ${nodeName}_end; i++) {
% if sizeB == 1:
        ${C}[i] = ${A}[i] * ${nodeName}_scalar;
% else:
        ${C}[i] = ${A}[i] * ${B}[i];
% endif
    }
}
""")
