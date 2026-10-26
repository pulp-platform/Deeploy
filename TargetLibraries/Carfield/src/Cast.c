/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Cast.h"

void Carfield_CastInt64ToInt32(int64_t *input, int32_t *output, int32_t size,
                               int32_t core_id, int32_t numThreads) {

  int32_t chunk = (size + numThreads - 1) / numThreads;
  int32_t start = core_id * chunk;
  int32_t end = start + chunk;
  if (end > size)
    end = size;
  if (start > size)
    start = size;

  for (int32_t i = start; i < end; i++) {
    output[i] = (int32_t)input[i];
  }
}
