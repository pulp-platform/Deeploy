# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
// Adam UpdateW - Weight Update (Name: ${nodeName}, Op: ${nodeOp})
BEGIN_SINGLE_CORE
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
    for (uint32_t ${nodeName}_i = 0; ${nodeName}_i < ${size}; ${nodeName}_i++) {
        ${W_new}[${nodeName}_i] = (1.0f - ${nodeName}_norm_coef_post) * (${X}[${nodeName}_i] - ${nodeName}_R_adjusted * ${V_new}[${nodeName}_i] / (sqrtf(${H_new}[${nodeName}_i]) + ${nodeName}_epsilon));
    }
END_SINGLE_CORE
""")
