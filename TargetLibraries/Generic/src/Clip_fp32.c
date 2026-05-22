/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Clip_fp32_fp32(float32_t *data_in, float32_t *data_out, float32_t min_val,
                    float32_t max_val, int32_t size) {
  for (int i = 0; i < size; i++) {
    data_out[i] = fmaxf(min_val, fminf(max_val, data_in[i]));
  }
}
