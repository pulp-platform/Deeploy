# SPDX-FileCopyrightText: 2021 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import NodeTemplate

referenceTemplate = NodeTemplate("""
BEGIN_SINGLE_CORE
    // SoftmaxCrossEntropyLoss (Name: ${nodeName}, Op: ${nodeOp})
    for (uint32_t i = 0; i < ${batch}; i++) {
        float max_logit = ${logits}[i * ${num_classes} + 0];
        for (uint32_t j = 1; j < ${num_classes}; j++) {
            if (${logits}[i * ${num_classes} + j] > max_logit) {
                max_logit = ${logits}[i * ${num_classes} + j];
            }
        }

        float32_t sum_exp = 0.0f;
        for (uint32_t j = 0; j < ${num_classes}; j++) {
            sum_exp += expf(${logits}[i * ${num_classes} + j] - max_logit);
        }

        for (uint32_t j = 0; j < ${num_classes}; j++) {
            // log_prob = logit - max_logit - log(sum_exp)
            ${log_prob}[i * ${num_classes} + j] = ${logits}[i * ${num_classes} + j] - max_logit - logf(sum_exp);
        }
    }
END_SINGLE_CORE
""")

referenceDualOutputTemplate = NodeTemplate("""
BEGIN_SINGLE_CORE
    // SoftmaxCrossEntropyLoss dual-output (Name: ${nodeName}, Op: ${nodeOp})
    float32_t sce_total_loss = 0.0f;
    for (uint32_t i = 0; i < ${batch}; i++) {
        float32_t sce_max_logit = ${logits}[i * ${num_classes}];
        for (uint32_t j = 1; j < ${num_classes}; j++) {
            if (${logits}[i * ${num_classes} + j] > sce_max_logit)
                sce_max_logit = ${logits}[i * ${num_classes} + j];
        }
        float32_t sce_sum_exp = 0.0f;
        for (uint32_t j = 0; j < ${num_classes}; j++)
            sce_sum_exp += expf(${logits}[i * ${num_classes} + j] - sce_max_logit);
        float32_t sce_log_sum_exp = logf(sce_sum_exp);
        for (uint32_t j = 0; j < ${num_classes}; j++)
            ${log_prob}[i * ${num_classes} + j] =
                ${logits}[i * ${num_classes} + j] - sce_max_logit - sce_log_sum_exp;
        sce_total_loss += -(${logits}[i * ${num_classes} + (uint32_t)(${labels}[i])]
                            - sce_max_logit - sce_log_sum_exp);
    }
    ${loss}[0] = sce_total_loss / (float32_t)${batch};
    printf("    [SCE] loss=%.6f\\r\\n", (double)${loss}[0]);
END_SINGLE_CORE
""")

referenceGradientTemplate = NodeTemplate("""
BEGIN_SINGLE_CORE
    // SoftmaxCrossEntropyLossGrad (Name: ${nodeName}, Op: ${nodeOp})
    float32_t batch_norm = 1.0f / ${total_batch};
    for (uint32_t i = 0; i < ${batch}; i++) {
        for (uint32_t j = 0; j < ${num_classes}; j++) {
            float32_t prob = expf(${log_prob}[i * ${num_classes} + j]);
            if (j == (${labels}[i])) {
                ${grad}[i * ${num_classes} + j] = (prob - 1.0f) * batch_norm * batch_norm; // RW: one batch_norm for loss norm, one for gradient norm
            } else {
                ${grad}[i * ${num_classes} + j] = prob * batch_norm * batch_norm;
            }
        }
    }

END_SINGLE_CORE
""")
