/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void RQiHardswish_s8_s8(int8_t *input, int8_t *output, int32_t size,
                        int32_t one_over_six, int32_t three, int32_t six,
                        int32_t input_offset, int32_t output_offset,
                        int32_t mul, int32_t add, int32_t shift) {

  int32_t temp;

  for (int i = 0; i < size; i++) {
    temp = input[i] + input_offset + three;
    if (temp < 0) {
      temp = 0;
    }
    if (temp > six) {
      temp = six;
    }
    temp = temp * one_over_six;
    temp = input[i] * temp;
    temp = temp * (mul) + (add);
    temp = ((temp + ((1 << ((shift)-1)))) >> (shift)) + output_offset;
    output[i] = (int8_t)CLAMP(temp, -128, 127);
  }
}
