# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Adam UpdateH - Second Moment Update, Parallel with 6x unrolling (Name: ${nodeName}, Op: ${nodeOp})
float32_t ${nodeName}_beta_coeff = ${beta};
float32_t ${nodeName}_norm_coef = ${norm_coefficient};
float32_t ${nodeName}_one_minus_beta = 1.0f - ${nodeName}_beta_coeff;

uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});

uint32_t i = ${nodeName}_chunk_start;
for (; i + 5 < ${nodeName}_chunk_stop; i += 6) {
    float32_t ${nodeName}_G_reg_0 = ${nodeName}_norm_coef * ${X}[i+0] + ${G}[i+0];
    float32_t ${nodeName}_G_reg_1 = ${nodeName}_norm_coef * ${X}[i+1] + ${G}[i+1];
    float32_t ${nodeName}_G_reg_2 = ${nodeName}_norm_coef * ${X}[i+2] + ${G}[i+2];
    float32_t ${nodeName}_G_reg_3 = ${nodeName}_norm_coef * ${X}[i+3] + ${G}[i+3];
    float32_t ${nodeName}_G_reg_4 = ${nodeName}_norm_coef * ${X}[i+4] + ${G}[i+4];
    float32_t ${nodeName}_G_reg_5 = ${nodeName}_norm_coef * ${X}[i+5] + ${G}[i+5];

    ${H_new}[i+0] = ${nodeName}_beta_coeff * ${H}[i+0] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg_0 * ${nodeName}_G_reg_0;
    ${H_new}[i+1] = ${nodeName}_beta_coeff * ${H}[i+1] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg_1 * ${nodeName}_G_reg_1;
    ${H_new}[i+2] = ${nodeName}_beta_coeff * ${H}[i+2] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg_2 * ${nodeName}_G_reg_2;
    ${H_new}[i+3] = ${nodeName}_beta_coeff * ${H}[i+3] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg_3 * ${nodeName}_G_reg_3;
    ${H_new}[i+4] = ${nodeName}_beta_coeff * ${H}[i+4] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg_4 * ${nodeName}_G_reg_4;
    ${H_new}[i+5] = ${nodeName}_beta_coeff * ${H}[i+5] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg_5 * ${nodeName}_G_reg_5;
}

for (; i < ${nodeName}_chunk_stop; i++) {
    float32_t ${nodeName}_G_reg = ${nodeName}_norm_coef * ${X}[i] + ${G}[i];
    ${H_new}[i] = ${nodeName}_beta_coeff * ${H}[i] + ${nodeName}_one_minus_beta * ${nodeName}_G_reg * ${nodeName}_G_reg;
}
""")
