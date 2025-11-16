/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Div_fp32_fp32_fp32(float32_t *data_in_1, float32_t *data_in_2,
                        float32_t *data_out, int32_t size) {
  for (int i = 0; i < size; i++) {
    data_out[i] = data_in_1[i] / data_in_2[i];
  }
}