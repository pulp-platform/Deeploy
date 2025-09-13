/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"

void RQiHardswish_s8_s8_plp(int8_t *input, int8_t *output, int32_t size,
                            int32_t one_over_six, int32_t three, int32_t six,
                            int32_t mul, int32_t add, int32_t shift) {

  int32_t temp;
  int32_t rnd;

  rnd = (1 << (shift - 1));

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, size);
  int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);

#pragma unroll 2
  for (int i = chunk_start; i < chunk_stop; i++) {
    temp = input[i] + three;
    temp = CLAMP(temp, 0, six);

    temp = temp * one_over_six;
    temp = input[i] * temp;
    temp = temp * (mul) + (add + rnd);

    temp = temp >> shift;

    output[i] = (int8_t)CLAMP(temp, -128, 127);
  }
}
