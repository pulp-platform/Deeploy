/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         float32_t *scale, float32_t *bias, float32_t epsilon,
                         int32_t size, int32_t lastDimLength) {
  float32_t mean;
  float32_t sum;
  float32_t std;
  float32_t temp;

  for (int i = 0; i < (size / lastDimLength); i++) {
    sum = 0.0f;
    mean = 0.0f;
    for (int j = 0; j < lastDimLength; j++) {
      mean += data_in[j + i * lastDimLength];
    }
    mean = mean / (float32_t)lastDimLength;
    for (int j = 0; j < lastDimLength; j++) {
      temp = data_in[j + i * lastDimLength] - mean;
      sum += temp * temp;
    }
    sum = sum / (float32_t)lastDimLength;
    sum += epsilon;
    std = sqrtf(sum);

    for (int j = 0; j < lastDimLength; j++) {
      data_out[j + i * lastDimLength] =
          ((data_in[j + i * lastDimLength] - mean) / std) * scale[j] + bias[j];
    }
  }
}
