# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Adam UpdateW - Weight Update, Parallel with 6x unrolling (Name: ${nodeName}, Op: ${nodeOp})
float32_t ${nodeName}_R_val = *${R};
int32_t ${nodeName}_T_val = *${T};
float32_t ${nodeName}_alpha = ${alpha};
float32_t ${nodeName}_beta_coeff = ${beta};
float32_t ${nodeName}_epsilon = ${epsilon};
float32_t ${nodeName}_norm_coef_post = ${norm_coefficient_post};
float32_t ${nodeName}_R_adjusted;
if (${nodeName}_T_val > 0) {
    ${nodeName}_R_adjusted = ${nodeName}_R_val * sqrtf(1.0f - powf(${nodeName}_beta_coeff, (float32_t)${nodeName}_T_val)) / (1.0f - powf(${nodeName}_alpha, (float32_t)${nodeName}_T_val));
} else {
    ${nodeName}_R_adjusted = ${nodeName}_R_val;
}
float32_t ${nodeName}_weight_decay = 1.0f - ${nodeName}_norm_coef_post;

uint8_t ${nodeName}_core_id = (uint8_t) pi_core_id();
uint8_t ${nodeName}_log2Core = (uint8_t) log2(NUM_CORES);
uint32_t ${nodeName}_chunk = (${size} >> ${nodeName}_log2Core) + ((${size} & (NUM_CORES-1))!=0);
uint32_t ${nodeName}_chunk_start = (uint32_t) MIN(${nodeName}_chunk*${nodeName}_core_id, (uint32_t) ${size});
uint32_t ${nodeName}_chunk_stop = (uint32_t) MIN(${nodeName}_chunk_start + ${nodeName}_chunk, (uint32_t) ${size});

uint32_t i = ${nodeName}_chunk_start;
for (; i + 5 < ${nodeName}_chunk_stop; i += 6) {
    ${W_new}[i+0] = ${nodeName}_weight_decay * (${X}[i+0] - ${nodeName}_R_adjusted * ${V_new}[i+0] / (sqrtf(${H_new}[i+0]) + ${nodeName}_epsilon));
    ${W_new}[i+1] = ${nodeName}_weight_decay * (${X}[i+1] - ${nodeName}_R_adjusted * ${V_new}[i+1] / (sqrtf(${H_new}[i+1]) + ${nodeName}_epsilon));
    ${W_new}[i+2] = ${nodeName}_weight_decay * (${X}[i+2] - ${nodeName}_R_adjusted * ${V_new}[i+2] / (sqrtf(${H_new}[i+2]) + ${nodeName}_epsilon));
    ${W_new}[i+3] = ${nodeName}_weight_decay * (${X}[i+3] - ${nodeName}_R_adjusted * ${V_new}[i+3] / (sqrtf(${H_new}[i+3]) + ${nodeName}_epsilon));
    ${W_new}[i+4] = ${nodeName}_weight_decay * (${X}[i+4] - ${nodeName}_R_adjusted * ${V_new}[i+4] / (sqrtf(${H_new}[i+4]) + ${nodeName}_epsilon));
    ${W_new}[i+5] = ${nodeName}_weight_decay * (${X}[i+5] - ${nodeName}_R_adjusted * ${V_new}[i+5] / (sqrtf(${H_new}[i+5]) + ${nodeName}_epsilon));
}

for (; i < ${nodeName}_chunk_stop; i++) {
    ${W_new}[i] = ${nodeName}_weight_decay * (${X}[i] - ${nodeName}_R_adjusted * ${V_new}[i] / (sqrtf(${H_new}[i]) + ${nodeName}_epsilon));
}
""")
