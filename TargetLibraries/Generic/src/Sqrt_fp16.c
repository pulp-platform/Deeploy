/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Sqrt_fp16_fp16(float16_t *data_in, float16_t *data_out, int32_t size) {
  for (int i = 0; i < size; i++) {
    data_out[i] = sqrtf(data_in[i]);
  }
}
