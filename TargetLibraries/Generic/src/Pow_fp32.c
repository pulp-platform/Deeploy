/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Pow_fp32_int32_fp32(float32_t *data_in, int32_t exponent,
                         float32_t *data_out, int32_t size) {
  for (int i = 0; i < size; i++) {
    float32_t result = 1.0f;
    int32_t exp = exponent;
    float32_t base = data_in[i];

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
