# ----------------------------------------------------------------------
#
# File: SoftmaxCrossEntropyTemplate.py
#
# Last edited: 09.03.2025
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author: Run Wang, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
