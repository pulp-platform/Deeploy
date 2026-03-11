/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

#define M_PI 3.14159265358979323846

void GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
  for (int i = 0; i < dataSize; i++) {
    float32_t x = data_in[i];
    float32_t cdf = 0.5f * (1.0f + tanhf((sqrtf(2.0f / (float)M_PI) *
                                          (x + 0.044715f * powf(x, 3.0f)))));
    data_out[i] = x * cdf;
  }
}

void GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out,
                            int32_t dataSize) {

  const float32_t scale = 1.702f;
  for (int i = 0; i < dataSize; i++) {
    float32_t x = data_in[i];
    float32_t sigmoid_in = scale * x;
    // sigmoid(z) = 1 / (1 + exp(-z))
    float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
    data_out[i] = x * sigmoid;
  }
}

void GELU_fp32_fp32_sigmoid_grad_chunk(float32_t *grad_in, float32_t *data_in,
                                       float32_t *grad_out, int32_t start_idx,
                                       int32_t end_idx) {
  // Exact GELU gradient: gelu'(x) = 0.5*(1+erf(x/sqrt(2))) + x*exp(-0.5*x^2)/sqrt(2*pi)
  // 1/sqrt(2)  = 0.70710678f
  // 1/sqrt(2*pi) = 0.39894228f
  for (int32_t i = start_idx; i < end_idx; i++) {
    float x = data_in[i];
    float gelu_derivative = 0.5f * (1.0f + erff(x * 0.70710678f)) +
                            x * expf(-0.5f * x * x) * 0.39894228f;
    grad_out[i] = grad_in[i] * gelu_derivative;
  }
}
