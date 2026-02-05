/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Softmax_fp32_fp32(float32_t *input, float32_t *output, int32_t size,
                       int32_t last_dim_length) {

  int32_t batch_size = size / last_dim_length;

  for (int b = 0; b < batch_size; b++) {
    float32_t max_val = -inf;
    float sum = 0.0f;

    for (int i = 0; i < last_dim_length; i++) {
      if (input[b * last_dim_length + i] > max_val) {
        max_val = input[b * last_dim_length + i];
      }
    }

    for (int i = 0; i < last_dim_length; i++) {
      float32_t exp_val = input[b * last_dim_length + i] - max_val;
      output[b * last_dim_length + i] = expf(exp_val);
      sum += output[b * last_dim_length + i];
    }

    for (int i = 0; i < last_dim_length; i++) {
      float32_t sum_1 = 1 / sum;
      output[b * last_dim_length + i] = output[b * last_dim_length + i] * sum_1;
    }
  }
}

void SoftmaxGrad_fp32_fp32_fp32(float32_t *upstream_grad,
                                float32_t *softmax_output,
                                float32_t *softmax_gradient, int32_t size,
                                int32_t last_dim_length) {

  int32_t batch_size = size / last_dim_length;

  for (int b = 0; b < batch_size; b++) {

    float32_t weighted_sum = 0.0f;

    for (int i = 0; i < last_dim_length; i++) {
      int idx = b * last_dim_length + i;
      weighted_sum += upstream_grad[idx] * softmax_output[idx];
    }

    for (int i = 0; i < last_dim_length; i++) {
      int idx = b * last_dim_length + i;
      softmax_gradient[idx] =
          softmax_output[idx] * (upstream_grad[idx] - weighted_sum);
    }
  }
}