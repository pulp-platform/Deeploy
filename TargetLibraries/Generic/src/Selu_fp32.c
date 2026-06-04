/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Selu_fp32_fp32(const float32_t *input, float32_t *output, int32_t size,
                    float32_t alpha, float32_t gamma) {

  for (int i = 0; i < size; i++) {
    float32_t tmp = input[i];
    if (input[i] < 0) {
      tmp = alpha * (expf(tmp) - 1.0f);
    }
    output[i] = gamma * tmp;
  }
}