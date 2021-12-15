/* =====================================================================
 * Title:        iSoftmax.c
 * Description:
 *
 * $Date:        13.11.2023
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULPSoftmax_u8_u8(uint8_t *data_in, uint8_t *data_out,
                       uint32_t *lastDimBuffer, uint32_t size,
                       uint32_t lastDimLength, int32_t coeffB, int32_t coeffC,
                       int32_t log2) {
  uint8_t z;
  int16_t xTilde, p;
  uint32_t y_sum;
  uint8_t x_max;

  uint32_t intermediateResult;
  uint32_t chunk, offset;

  if (pi_core_id() < (NUM_CORES - 1)) {
    chunk = (size / lastDimLength) / NUM_CORES;
    offset = chunk * lastDimLength * pi_core_id();
    lastDimBuffer += lastDimLength * pi_core_id();
  } else {
    uint32_t prevChunk = (size / lastDimLength) / NUM_CORES;
    chunk = (size / lastDimLength) - prevChunk * (NUM_CORES - 1);
    offset = size - (chunk * lastDimLength);
    lastDimBuffer += lastDimLength * pi_core_id();
  }

  for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
       i += lastDimLength) {
    y_sum = 0;
    x_max = 0;
    for (uint32_t j = 0; j < lastDimLength; j++) {
      if (data_in[j + i] > x_max) {
        x_max = data_in[j + i];
      }
    }
    for (uint32_t j = 0; j < lastDimLength; j++) {
      xTilde = ((data_in[j + i]) - x_max);
      z = (uint8_t)(-(xTilde / log2));
      z = CLAMP(z, 0, 31);
      p = (xTilde + z * log2);
      intermediateResult = (uint32_t)(((p + coeffB) * (p + coeffB)) + coeffC);
      lastDimBuffer[j] = (uint32_t)(intermediateResult >> (z));
      y_sum += lastDimBuffer[j];
    }
    for (uint32_t j = 0; j < lastDimLength; j++) {
      data_out[j + i] = (uint8_t)((lastDimBuffer[j] * 255) / (y_sum));
    }
  }
}

void PULPSoftmax_i8_u8(int8_t *data_in, uint8_t *data_out,
                       uint32_t *lastDimBuffer, uint32_t size,
                       uint32_t lastDimLength, int32_t coeffB, int32_t coeffC,
                       int32_t log2) {
  uint8_t z;
  int16_t xTilde, p;
  uint32_t y_sum;
  int8_t x_max;

  uint32_t intermediateResult;
  uint32_t chunk, offset;

  if (pi_core_id() < (NUM_CORES - 1)) {
    chunk = (size / lastDimLength) / NUM_CORES;
    offset = chunk * lastDimLength * pi_core_id();
    lastDimBuffer += lastDimLength * pi_core_id();
  } else {
    uint32_t prevChunk = (size / lastDimLength) / NUM_CORES;
    chunk = (size / lastDimLength) - prevChunk * (NUM_CORES - 1);
    offset = size - (chunk * lastDimLength);
    lastDimBuffer += lastDimLength * pi_core_id();
  }

  for (uint32_t i = offset; i < offset + (chunk * lastDimLength);
       i += lastDimLength) {

    y_sum = 0;
    x_max = -128;
    for (uint32_t j = 0; j < lastDimLength; j++) {
      if (data_in[j + i] > x_max) {
        x_max = data_in[j + i];
      }
    }
    for (uint32_t j = 0; j < lastDimLength; j++) {
      xTilde = ((data_in[j + i]) - x_max);
      z = (uint8_t)(-(xTilde / log2));
      z = CLAMP(z, 0, 31);
      p = (xTilde + z * log2);
      intermediateResult = (((p + coeffB) * (p + coeffB)) + coeffC);
      lastDimBuffer[j] = (intermediateResult >> (z));
      y_sum += lastDimBuffer[j];
    }
    for (uint32_t j = 0; j < lastDimLength; j++) {
      data_out[j + i] = (uint8_t)((lastDimBuffer[j] * 255) / (y_sum));
    }
  }
}
