/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void HardSwish_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         int32_t size) {
  for (int i = 0; i < size; i++) {
    float32_t x = data_in[i];
    data_out[i] = x * fmaxf(0, fminf(1, x / 6 + 0.5));
  }
}
