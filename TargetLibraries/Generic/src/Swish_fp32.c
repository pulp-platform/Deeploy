/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Swish_fp32_fp32(float32_t *data_in, float32_t *data_out, float alpha,
                     int32_t size) {
  for (int i = 0; i < size; i++) {
    float32_t x = data_in[i];
    data_out[i] = x / (1 + expf(-alpha * x));
  }
}
