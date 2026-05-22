/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void InstanceNormalization_fp32_fp32(
    const float32_t *__restrict__ src, float32_t *__restrict__ dst,
    const float32_t *__restrict__ scale, const float32_t *__restrict__ bias,
    uint32_t batch_size, uint32_t num_channels,
    uint32_t spatial, // spatial dimension (L or H*W or D*H*W, etc.)
    float32_t epsilon) {

  if (spatial == 0) {
    return;
  }

  uint32_t slice = num_channels * spatial; // elements per batch

  for (uint32_t n = 0; n < batch_size; ++n) {
    for (uint32_t c = 0; c < num_channels; ++c) {
      uint32_t channel_offset = n * slice + c * spatial;
      const float32_t *x = src + channel_offset;
      float32_t *y = dst + channel_offset;

      /* --- mean --- */
      float64_t sum = 0.0;
      for (uint32_t i = 0; i < spatial; ++i)
        sum += x[i];
      float64_t mean = sum / (float32_t)spatial;

      /* --- variance --- */
      float64_t var = 0.0;
      for (uint32_t i = 0; i < spatial; ++i) {
        float64_t d = (float64_t)x[i] - mean;
        var += d * d;
      }
      var /= (float64_t)spatial;

      /* --- normalize + affine --- */
      float32_t inv_std = (float32_t)(1.0 / sqrt(var + (float64_t)epsilon));
      float32_t g = scale[c];
      float32_t b = bias[c];
      float32_t m = (float32_t)mean;

      for (size_t i = 0; i < spatial; ++i) {
        y[i] = g * (x[i] - m) * inv_std + b;
      }
    }
  }
}
