// SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeployBasicMath.h"
#include <math.h>

void BatchNorm_fp32(const float32_t *input, const float32_t *gamma,
                    const float32_t *beta, const float32_t *mean,
                    const float32_t *var, float32_t *output, int N, int C,
                    int L, float epsilon, int channels_first) {
#pragma omp parallel for
  for (int c = 0; c < C; ++c) {
    float32_t c_mean = mean[c];
    float32_t c_var = var[c];
    float32_t c_gamma = gamma[c];
    float32_t c_beta = beta[c];
    float32_t denom = sqrtf(c_var + epsilon);
    for (int n = 0; n < N; ++n) {
      for (int l = 0; l < L; ++l) {
        int index;
        if (channels_first) {
          index = n * C * L + c * L + l;
        } else {
          index = n * C * L + l * C + c;
        }
        float32_t x = input[index];
        float32_t norm = (x - c_mean) / denom;
        output[index] = c_gamma * norm + c_beta;
      }
    }
  }
}
