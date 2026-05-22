/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void GroupNormalization_fp32_fp32(
    const float32_t *__restrict__ src, float32_t *__restrict__ dst,
    const float32_t *__restrict__ scale, const float32_t *__restrict__ bias,
    uint32_t batch_size, uint32_t num_channels,
    uint32_t spatial, // spatial dimension (L or H*W or D*H*W, etc.)
    uint32_t num_groups, float32_t epsilon) {

  if (num_groups == 0 || spatial == 0 || (num_channels % num_groups) != 0) {
    return;
  }
  uint32_t channels_per_group = num_channels / num_groups;
  uint32_t group_elements = channels_per_group * spatial;
  if (group_elements == 0) {
    return;
  }
  uint32_t slice = num_channels * spatial; // elements per batch

  for (uint32_t n = 0; n < batch_size; ++n) {
    for (uint32_t g = 0; g < num_groups; ++g) {
      uint32_t group_offset = n * slice + g * group_elements;
      const float32_t *x_group = src + group_offset;

      /* --- mean --- */
      float64_t sum = 0.0;
      for (uint32_t i = 0; i < group_elements; ++i) {
        sum += x_group[i];
      }
      float64_t mean = sum / (float32_t)group_elements;

      /* --- variance --- */
      float64_t var = 0.0;
      for (uint32_t i = 0; i < group_elements; ++i) {
        float64_t d = (float64_t)x_group[i] - mean;
        var += d * d;
      }
      var /= (float64_t)group_elements;

      /* --- normalize + affine --- */
      float32_t inv_std = (float32_t)(1.0 / sqrt(var + (float64_t)epsilon));
      float32_t m = (float32_t)mean;

      for (uint32_t lc = 0; lc < channels_per_group; ++lc) {
        const float32_t *x_channel = x_group + lc * spatial;
        float32_t *y_channel = dst + group_offset + lc * spatial;
        uint32_t c = g * channels_per_group + lc; // global channel
        float32_t s = scale[c];
        float32_t b = bias[c];

        for (uint32_t i = 0; i < spatial; ++i) {
          y_channel[i] = s * (x_channel[i] - m) * inv_std + b;
        }
      }
    }
  }
}
