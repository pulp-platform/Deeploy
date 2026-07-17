# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
BEGIN_SINGLE_CORE
    float32_t R_val = *${R};
    // Note: ONNX spec defines T as int64, but we use int32 for embedded compatibility
    int32_t T_val = *${T};
    float32_t alpha = ${alpha};
    float32_t beta_coeff = ${beta};
    float32_t epsilon = ${epsilon};
    float32_t norm_coef = ${norm_coefficient};
    float32_t norm_coef_post = ${norm_coefficient_post};
    float32_t R_adjusted;
    if (T_val > 0) {
        R_adjusted = R_val * sqrtf(1.0f - powf(beta_coeff, (float32_t)T_val)) / (1.0f - powf(alpha, (float32_t)T_val));
    } else {
        R_adjusted = R_val;
    }
    for (uint32_t i = 0; i < ${size}; i++) {
        float32_t G_reg = norm_coef * ${X}[i] + ${G}[i];
        ${V}[i] = alpha * ${V}[i] + (1.0f - alpha) * G_reg;
        ${H}[i] = beta_coeff * ${H}[i] + (1.0f - beta_coeff) * G_reg * G_reg;
        ${X_new}[i] = (1.0f - norm_coef_post) * (${X}[i] - R_adjusted * ${V}[i] / (sqrtf(${H}[i]) + epsilon));
    }
END_SINGLE_CORE
""")
