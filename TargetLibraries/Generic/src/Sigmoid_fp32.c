/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Sigmoid_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t size) {
  for (int i = 0; i < size; i++) {
    data_out[i] = 1 / (1 + expf(-data_in[i]));
  }
}
