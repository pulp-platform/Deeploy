/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "ArgMax.h"

void Carfield_ArgMaxChannel(float32_t *input, int64_t *output, int32_t C,
                            int32_t spatialSize, int32_t core_id,
                            int32_t numThreads) {

  int32_t chunk = (spatialSize + numThreads - 1) / numThreads;
  int32_t start = core_id * chunk;
  int32_t end = start + chunk;
  if (end > spatialSize)
    end = spatialSize;
  if (start > spatialSize)
    start = spatialSize;

  for (int32_t i = start; i < end; i++) {
    float32_t best = input[i];
    int32_t bestIdx = 0;

    for (int32_t c = 1; c < C; c++) {
      float32_t v = input[c * spatialSize + i];
      if (v > best) {
        best = v;
        bestIdx = c;
      }
    }

    output[i] = (int64_t)bestIdx;
  }
}
