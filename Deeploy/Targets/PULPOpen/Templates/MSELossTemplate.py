# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
BEGIN_SINGLE_CORE
    // MSELoss (Name: ${nodeName}, Op: ${nodeOp})
    // loss = mean((pred - target)^2)
    float32_t mse_sum = 0.0f;
    for (uint32_t i = 0; i < ${num_elements}; i++) {
        float32_t mse_d = ${pred}[i] - ${target}[i];
        mse_sum += mse_d * mse_d;
    }
    ${loss}[0] = mse_sum / (float32_t)${num_elements};
    printf("    [MSE] loss=%.6f\\r\\n", (double)${loss}[0]);
END_SINGLE_CORE
""")

referenceGradientTemplate = NodeTemplate("""
BEGIN_SINGLE_CORE
    // MSELossGrad (Name: ${nodeName}, Op: ${nodeOp})
    // grad = 2 * (pred - target) / N
    float32_t mse_grad_scale = 2.0f / (float32_t)${num_elements};
    for (uint32_t i = 0; i < ${num_elements}; i++) {
        ${grad}[i] = mse_grad_scale * (${pred}[i] - ${target}[i]);
    }
END_SINGLE_CORE
""")
