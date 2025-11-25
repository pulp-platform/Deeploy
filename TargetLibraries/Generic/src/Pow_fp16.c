/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "DeeployBasicMath.h"

void Pow_fp16_int32_fp16(float16_t *data_in, int32_t exponent,
                         float16_t *data_out, int32_t size) {
  for (int i = 0; i < size; i++) {
    float16_t result = 1.0f;
    int32_t exp = exponent;
    float16_t base = data_in[i];

    if (exp < 0) {
      base = 1.0f / base;
      exp = -exp;
    }

    for (int32_t j = 0; j < exp; j++) {
      result *= base;
    }

    data_out[i] = result;
  }
}
