/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"
#include <stdint.h>

inline int16_t _plp_sqrt_q16(int16_t pSrc) {

  int16_t number = pSrc;
  int16_t root = 0;

  int16_t start = 0;
  int16_t end = 255; // smallest integer that is larger than sqrt(0x7FFF)
  int16_t mid;

  while (start <= end) {

    mid = (start + end) >> 1;

    if (((mid * mid)) == number) {
      root = mid;
      break;
    }

    if (((mid * mid)) < number) {
      start = mid + 1;
      root = mid;
    } else {
      end = mid - 1;
    }
  }

  return root;
}

void iRMSnorm_s8_s8_plp(int8_t *data_in, int8_t *data_out, int32_t *weight,
                        int32_t size, int32_t lastDimLength, int32_t log2D) {

  int32_t sum;
  int32_t std;
  int16_t temp, temp1;
  int32_t intermediate;

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk =
      (lastDimLength >> log2Core) + ((lastDimLength & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, lastDimLength);
  int16_t chunk_stop = MIN(chunk_start + chunk, lastDimLength + 1);

  for (int i = 0; i < (size / lastDimLength); i++) {
    sum = 0;

#pragma unroll 8
    for (int j = 0; j < lastDimLength; j++) {
      temp = (data_in[j + i * lastDimLength]);
      sum += temp * temp;
    }

    sum = sum / lastDimLength;
    sum += 1;
    std = _plp_sqrt_q16((int16_t)sum);

    for (int j = chunk_start; j < chunk_stop; j++) {

      intermediate =
          (((data_in[j + i * lastDimLength] * weight[j]) / (std)) >> log2D);

      data_out[j + i * lastDimLength] = CLAMP(intermediate, -128, 127);
    }
  }
}
