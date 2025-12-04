/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Pow_fp32_fp32_fp32(const float32_t *__restrict__ data_in,
                        const float32_t *__restrict__ exponent,
                        float32_t *__restrict__ data_out, int32_t size) {
  for (int i = 0; i < size; i++) {
    data_out[i] = powf(data_in[i], exponent[i]);
  }
}

void Pow_fp32_scalar_fp32(const float32_t *__restrict__ data_in,
                          float32_t exponent, float32_t *__restrict__ data_out,
                          int32_t size) {
  for (int i = 0; i < size; i++) {
    data_out[i] = powf(data_in[i], exponent);
  }
}
