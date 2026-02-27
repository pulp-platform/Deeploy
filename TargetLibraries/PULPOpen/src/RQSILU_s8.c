/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void RQSILU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize,
                  int32_t input_offset) {
  for (int i = 0; i < dataSize; i++) {
    int32_t x = data_in[i] + 128 - input_offset;
    data_out[i] = RQSILU_lut_s8_s8[x];
  }
}
