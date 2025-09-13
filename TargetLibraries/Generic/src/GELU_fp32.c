/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

#define M_PI 3.14159265358979323846

void GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
  for (int i = 0; i < dataSize; i++) {
    float32_t x = data_in[i];
    float32_t cdf = 0.5f * (1.0f + tanhf((sqrtf(2.0f / (float)M_PI) *
                                          (x + 0.044715f * powf(x, 3.0f)))));
    data_out[i] = x * cdf;
  }
}

void GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out,
                            int32_t dataSize) {

  const float32_t scale = 1.702f;
  for (int i = 0; i < dataSize; i++) {
    float32_t x = data_in[i];
    float32_t sigmoid_in = scale * x;
    // sigmoid(z) = 1 / (1 + exp(-z))
    float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
    data_out[i] = x * sigmoid;
  }
}
