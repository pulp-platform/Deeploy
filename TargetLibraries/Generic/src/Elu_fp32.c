/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Elu_fp32_fp32(const float32_t *input, float32_t *output, int32_t size,
                   float32_t alpha) {

  for (int i = 0; i < size; i++) {
    if (input[i] >= 0) {
      output[i] = input[i];
    } else {
      output[i] = alpha * (expf(input[i]) - 1.0f);
    }
  }
}