/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Relu_fp32_fp32(float32_t *input, float32_t *output, int32_t size) {

  for (int i = 0; i < size; i++) {
    output[i] = MAX(input[i], 0.0f);
  }
}
