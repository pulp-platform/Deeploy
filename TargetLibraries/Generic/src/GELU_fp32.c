/* =====================================================================
 * Title:        GELU_fp32.c
 * Description:
 *
 * $Date:        19.12.2022
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Run Wang, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployBasicMath.h"
#define M_PI 3.14159265358979323846

void GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
   for (int i = 0; i < dataSize; i++) {
        float32_t x = data_in[i];
        float32_t cdf = 0.5 * (1.0 + tanh((sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3)))));
        data_out[i] = x * cdf;
    }
}

void GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out, int32_t dataSize) {

    const float32_t scale = 1.702f;
    for (int i = 0; i < dataSize; i++) {
        float32_t x = data_in[i];
        float32_t sigmoid_in = scale * x;
        // sigmoid(z) = 1 / (1 + exp(-z))
        float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
        data_out[i] = x * sigmoid;
    }
}

void GELU_fp32_fp32_sigmoid_chunk(float32_t *data_in, float32_t *data_out, int32_t start_idx, int32_t end_idx)
{
    const float32_t scale = 1.702f;
    for (uint32_t i = start_idx; i < end_idx; i++)
    {
        float32_t x = data_in[i];
        float32_t sigmoid_in = scale * x;
        // sigmoid(z) = 1 / (1 + exp(-z))
        float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
        data_out[i] = x * sigmoid;
    }
}

void GELU_fp32_fp32_sigmoid_grad_chunk(float32_t *grad_in, float32_t *data_in, float32_t *grad_out, int32_t start_idx, int32_t end_idx)
{
    // d(Gelu)/dx ≈ sigmoid(1.702 * x) + x * sigmoid(1.702 * x) * (1 - sigmoid(1.702 * x)) * 1.702
    const float COEFF = 1.702f;
    for (int32_t i = start_idx; i < end_idx; i++) {
            float x = data_in[i];
            float upstream_grad = grad_in[i];
            
            // Gelu(x) = x * sigmoid(1.702 * x)
            
            // Compute sigmoid(1.702 * x)
            float z = COEFF * x;
            float sigmoid_z = 1.0f / (1.0f + expf(-z));
            
            // d(Gelu)/dx = sigmoid(1.702*x) + x * sigmoid(1.702*x) * (1-sigmoid(1.702*x)) * 1.702
            float sigmoid_derivative = sigmoid_z * (1.0f - sigmoid_z) * COEFF;
            float gelu_derivative = sigmoid_z + x * sigmoid_derivative;
            
            // The backward gradient is the product of the upstream gradient and the derivative
            grad_out[i] = upstream_grad * gelu_derivative;
        }
}
