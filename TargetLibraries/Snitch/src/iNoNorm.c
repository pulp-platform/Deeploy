/* ----------------------------------------------------------------------
#
# File: iNoNorm.c
#
# Last edited: 06.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include "DeeploySnitchMath.h"

void SnitchiNoNorm_s8_s8(int8_t *data_in, int8_t *data_out, int8_t *weights,
                         int32_t *bias, uint32_t size, int32_t mul,
                         int32_t log2D) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();
  uint32_t chunk, chunkSize, start, stop;

  chunkSize = size / numThreads;
  if (core_id < (numThreads - 1)) {
    chunk = chunkSize * core_id;
    stop = chunk + chunkSize;
  } else {
    chunk = (chunkSize * core_id - 1) + (size - chunk);
    stop = size;
  }
  start = chunk;

  uint32_t packedIn, packedWeights;
  int8_t unpackedIn1, unpackedIn2, unpackedIn3, unpackedIn4;
  int8_t unpackedWeights1, unpackedWeights2, unpackedWeights3, unpackedWeights4;
  int16_t partialProduct1, partialProduct2, partialProduct3, partialProduct4;

  int32_t *dataInPtr = (int32_t *)(data_in);
  int32_t *weightsPtr = (int32_t *)(weights);
  int32_t *outputPtr = (int32_t *)(data_out);

  uint32_t firstReminderLoopSize = start % 4;
  uint32_t lastReminderLoopSize = stop % 4;
  uint32_t firstReminderLoopIdx = start;
  uint32_t lastReminderLoopIdx = stop - lastReminderLoopSize;
  start = (start + firstReminderLoopSize) >> 2;
  stop = (stop - lastReminderLoopSize) >> 2;
  uint32_t biasIdx = start * 4;

  // JUNGVI: Compute sequentially the first elements not aligned to a word (32b)
  for (uint32_t i = firstReminderLoopIdx;
       i < firstReminderLoopIdx + firstReminderLoopSize; i++) {
    data_out[i] =
        ((((int32_t)data_in[i] * weights[i]) + bias[i]) * mul) >> log2D;
  }

  for (uint32_t i = start; i < stop; i++) {

    packedIn = dataInPtr[i];
    packedWeights = weightsPtr[i];

    unpackedIn1 = (packedIn & 0x000000FF);
    unpackedIn2 = (packedIn & 0x0000FF00) >> 8;
    unpackedIn3 = (packedIn & 0x00FF0000) >> 16;
    unpackedIn4 = packedIn >> 24;

    unpackedWeights1 = (packedWeights & 0x000000FF);
    unpackedWeights2 = (packedWeights & 0x0000FF00) >> 8;
    unpackedWeights3 = (packedWeights & 0x00FF0000) >> 16;
    unpackedWeights4 = packedWeights >> 24;

    partialProduct1 = (int16_t)(unpackedIn1 * unpackedWeights1);
    partialProduct2 = (int16_t)(unpackedIn2 * unpackedWeights2);
    partialProduct3 = (int16_t)(unpackedIn3 * unpackedWeights3);
    partialProduct4 = (int16_t)(unpackedIn4 * unpackedWeights4);

    uint8_t outBuf1 = ((partialProduct1 + bias[biasIdx + 0]) * mul) >> log2D;
    uint8_t outBuf2 = ((partialProduct2 + bias[biasIdx + 1]) * mul) >> log2D;
    uint8_t outBuf3 = ((partialProduct3 + bias[biasIdx + 2]) * mul) >> log2D;
    uint8_t outBuf4 = ((partialProduct4 + bias[biasIdx + 3]) * mul) >> log2D;

    uint32_t outPacked =
        (outBuf1 << 0) | (outBuf2 << 8) | (outBuf3 << 16) | (outBuf4 << 24);
    outputPtr[i] = outPacked;
    biasIdx += 4;
  }

  // JUNGVI: Compute sequentially the last elements not aligned to a word (32b)
  for (uint32_t i = lastReminderLoopIdx;
       i < lastReminderLoopIdx + lastReminderLoopSize; i++) {
    data_out[i] =
        ((((int32_t)data_in[i] * weights[i]) + bias[i]) * mul) >> log2D;
  }
}
