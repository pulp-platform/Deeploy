/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void iHardswish_s8_s32(int8_t *input, int32_t *output, int32_t size,
                       int32_t one_over_six, int32_t three, int32_t six,
                       int32_t input_offset) {

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
    output[i] = temp;
  }
}