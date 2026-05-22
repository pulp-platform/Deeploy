/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void GlobalAveragePool_fp32_fp32(float32_t const *__restrict__ src,
                                 float32_t *__restrict__ dst, uint32_t N,
                                 uint32_t C, uint32_t spatial_size) {

  if (spatial_size == 0) {
    return; // invalid shape for average pooling; avoid divide-by-zero
  }
  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t c = 0; c < C; ++c) {

      float32_t sum = 0.0f;
      const float32_t *x = src + (n * C + c) * spatial_size;

      for (uint32_t i = 0; i < spatial_size; ++i) {
        sum += x[i];
      }
      dst[n * C + c] = sum / spatial_size;
    }
  }
}