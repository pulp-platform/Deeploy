/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void GELU_s8_s32(int8_t *data_in, int32_t *data_out, int32_t dataSize, int8_t b,
                 int16_t one, int32_t input_offset) {
  int32_t sign, x, x_abs, q;
  int32_t d;
  int32_t L, y;
  for (int i = 0; i < dataSize; i++) {
    x = data_in[i] + input_offset;
    sign = (x > 0) - (x < 0); // sgn(x)
    x_abs = sign * x;         // abs(x)
    if (x_abs > -b) {
      q = -b;
    } else {
      q = x_abs;
    }
    d = q + b;
    L = sign * (-(d * d) + one);
    y = x * (((one + L)) >> 1);
    data_out[i] = y;
  }
}
