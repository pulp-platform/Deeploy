/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         float32_t *scale, float32_t *bias, float32_t epsilon,
                         int32_t size, int32_t lastDimLength) {
  float32_t mean;
  float32_t sum;
  float32_t std;
  float32_t temp;

  for (int i = 0; i < (size / lastDimLength); i++) {
    sum = 0.0f;
    mean = 0.0f;
    for (int j = 0; j < lastDimLength; j++) {
      mean += data_in[j + i * lastDimLength];
    }
    mean = mean / (float32_t)lastDimLength;
    for (int j = 0; j < lastDimLength; j++) {
      temp = data_in[j + i * lastDimLength] - mean;
      sum += temp * temp;
    }
    sum = sum / (float32_t)lastDimLength;
    sum += epsilon;
    std = sqrtf(sum);

    for (int j = 0; j < lastDimLength; j++) {
      data_out[j + i * lastDimLength] =
          ((data_in[j + i * lastDimLength] - mean) / std) * scale[j] + bias[j];
    }
  }
}

void LayernormGrad_fp32_fp32(float32_t *grad_in, float32_t *data_in,
                             float32_t *grad_out, float32_t *scale,
                             float32_t epsilon, int32_t size,
                             int32_t lastDimLength) {
  float32_t mean, variance, std, inv_std;
  float32_t sum_dy, sum_dy_scaled_centered;
  float32_t centered_input;

  for (int i = 0; i < (size / lastDimLength); i++) {
    // Step 1: Recompute mean and variance from forward pass
    mean = 0.0f;
    variance = 0.0f;

    for (int j = 0; j < lastDimLength; j++) {
      mean += data_in[j + i * lastDimLength];
    }
    mean = mean / lastDimLength;

    for (int j = 0; j < lastDimLength; j++) {
      centered_input = data_in[j + i * lastDimLength] - mean;
      variance += centered_input * centered_input;
    }
    variance = variance / lastDimLength;
    variance += epsilon;
    std = sqrtf(variance);
    inv_std = 1.0f / std;

    // Step 2: Compute intermediate values needed for gradient calculation
    sum_dy = 0.0f;
    sum_dy_scaled_centered = 0.0f;

    for (int j = 0; j < lastDimLength; j++) {
      sum_dy += grad_in[j + i * lastDimLength];
      centered_input = data_in[j + i * lastDimLength] - mean;
      sum_dy_scaled_centered +=
          grad_in[j + i * lastDimLength] * scale[j] * centered_input * inv_std;
    }

    // Step 3: Calculate dX gradient for each element
    for (int j = 0; j < lastDimLength; j++) {
      centered_input = data_in[j + i * lastDimLength] - mean;

      // dx = (1/std) * scale * (dy - (1/N)*sum(dy) -
      // (x-mean)/(N*std^2)*sum(dy*scale*(x-mean)/std))
      grad_out[j + i * lastDimLength] =
          inv_std * scale[j] *
          (grad_in[j + i * lastDimLength] - (sum_dy / lastDimLength) -
           (centered_input * inv_std * inv_std / lastDimLength) *
               sum_dy_scaled_centered);
    }
  }
}

void LayernormGradParam_fp32_fp32(float32_t *grad_in, float32_t *data_in,
                                  float32_t *weight_grad, float32_t *bias_grad,
                                  float32_t epsilon, int32_t size,
                                  int32_t lastDimLength) {
  float32_t mean, variance, std, inv_std;
  float32_t centered_input, hat_x;
  int32_t num_sequences = size / lastDimLength;

  // Initialize output gradients to zero
  for (int j = 0; j < lastDimLength; j++) {
    weight_grad[j] = 0.0f;
    bias_grad[j] = 0.0f;
  }

  for (int i = 0; i < num_sequences; i++) {
    // Recompute mean and variance from forward pass
    mean = 0.0f;
    for (int j = 0; j < lastDimLength; j++) {
      mean += data_in[j + i * lastDimLength];
    }
    mean = mean / lastDimLength;

    variance = 0.0f;
    for (int j = 0; j < lastDimLength; j++) {
      centered_input = data_in[j + i * lastDimLength] - mean;
      variance += centered_input * centered_input;
    }
    variance = variance / lastDimLength;
    variance += epsilon;
    std = sqrtf(variance);
    inv_std = 1.0f / std;

    // Accumulate dscale and dbias over sequences
    for (int j = 0; j < lastDimLength; j++) {
      hat_x = (data_in[j + i * lastDimLength] - mean) * inv_std;
      weight_grad[j] += grad_in[j + i * lastDimLength] * hat_x;
      bias_grad[j] += grad_in[j + i * lastDimLength];
    }
  }
}
