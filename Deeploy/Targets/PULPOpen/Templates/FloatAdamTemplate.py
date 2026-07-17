# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
float32_t ${nodeName}_R_val = *${R};
int32_t ${nodeName}_T_val = *${T};
float32_t ${nodeName}_alpha = ${alpha};
float32_t ${nodeName}_beta_coeff = ${beta};
float32_t ${nodeName}_epsilon = ${epsilon};
float32_t ${nodeName}_norm_coef = ${norm_coefficient};
float32_t ${nodeName}_norm_coef_post = ${norm_coefficient_post};
float32_t ${nodeName}_R_adjusted;
if (${nodeName}_T_val > 0) {
    ${nodeName}_R_adjusted = ${nodeName}_R_val * sqrtf(1.0f - powf(${nodeName}_beta_coeff, (float32_t)${nodeName}_T_val)) / (1.0f - powf(${nodeName}_alpha, (float32_t)${nodeName}_T_val));
} else {
    ${nodeName}_R_adjusted = ${nodeName}_R_val;
}

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

    ${V}[i+0] = ${nodeName}_alpha * ${V}[i+0] + (1.0f - ${nodeName}_alpha) * ${nodeName}_G_reg_0;
    ${V}[i+1] = ${nodeName}_alpha * ${V}[i+1] + (1.0f - ${nodeName}_alpha) * ${nodeName}_G_reg_1;
    ${V}[i+2] = ${nodeName}_alpha * ${V}[i+2] + (1.0f - ${nodeName}_alpha) * ${nodeName}_G_reg_2;
    ${V}[i+3] = ${nodeName}_alpha * ${V}[i+3] + (1.0f - ${nodeName}_alpha) * ${nodeName}_G_reg_3;
    ${V}[i+4] = ${nodeName}_alpha * ${V}[i+4] + (1.0f - ${nodeName}_alpha) * ${nodeName}_G_reg_4;
    ${V}[i+5] = ${nodeName}_alpha * ${V}[i+5] + (1.0f - ${nodeName}_alpha) * ${nodeName}_G_reg_5;

    ${H}[i+0] = ${nodeName}_beta_coeff * ${H}[i+0] + (1.0f - ${nodeName}_beta_coeff) * ${nodeName}_G_reg_0 * ${nodeName}_G_reg_0;
    ${H}[i+1] = ${nodeName}_beta_coeff * ${H}[i+1] + (1.0f - ${nodeName}_beta_coeff) * ${nodeName}_G_reg_1 * ${nodeName}_G_reg_1;
    ${H}[i+2] = ${nodeName}_beta_coeff * ${H}[i+2] + (1.0f - ${nodeName}_beta_coeff) * ${nodeName}_G_reg_2 * ${nodeName}_G_reg_2;
    ${H}[i+3] = ${nodeName}_beta_coeff * ${H}[i+3] + (1.0f - ${nodeName}_beta_coeff) * ${nodeName}_G_reg_3 * ${nodeName}_G_reg_3;
    ${H}[i+4] = ${nodeName}_beta_coeff * ${H}[i+4] + (1.0f - ${nodeName}_beta_coeff) * ${nodeName}_G_reg_4 * ${nodeName}_G_reg_4;
    ${H}[i+5] = ${nodeName}_beta_coeff * ${H}[i+5] + (1.0f - ${nodeName}_beta_coeff) * ${nodeName}_G_reg_5 * ${nodeName}_G_reg_5;

    ${X_new}[i+0] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[i+0] - ${nodeName}_R_adjusted * ${V}[i+0] / (sqrtf(${H}[i+0]) + ${nodeName}_epsilon));
    ${X_new}[i+1] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[i+1] - ${nodeName}_R_adjusted * ${V}[i+1] / (sqrtf(${H}[i+1]) + ${nodeName}_epsilon));
    ${X_new}[i+2] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[i+2] - ${nodeName}_R_adjusted * ${V}[i+2] / (sqrtf(${H}[i+2]) + ${nodeName}_epsilon));
    ${X_new}[i+3] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[i+3] - ${nodeName}_R_adjusted * ${V}[i+3] / (sqrtf(${H}[i+3]) + ${nodeName}_epsilon));
    ${X_new}[i+4] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[i+4] - ${nodeName}_R_adjusted * ${V}[i+4] / (sqrtf(${H}[i+4]) + ${nodeName}_epsilon));
    ${X_new}[i+5] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[i+5] - ${nodeName}_R_adjusted * ${V}[i+5] / (sqrtf(${H}[i+5]) + ${nodeName}_epsilon));
}

for (; i < ${nodeName}_chunk_stop; i++) {
    float32_t ${nodeName}_G_reg = ${nodeName}_norm_coef * ${X}[i] + ${G}[i];
    ${V}[i] = ${nodeName}_alpha * ${V}[i] + (1.0f - ${nodeName}_alpha) * ${nodeName}_G_reg;
    ${H}[i] = ${nodeName}_beta_coeff * ${H}[i] + (1.0f - ${nodeName}_beta_coeff) * ${nodeName}_G_reg * ${nodeName}_G_reg;
    ${X_new}[i] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[i] - ${nodeName}_R_adjusted * ${V}[i] / (sqrtf(${H}[i]) + ${nodeName}_epsilon));
}
""")
