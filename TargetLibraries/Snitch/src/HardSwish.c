/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

void HardSwish_fp32(float32_t *data_in, float32_t *data_out, uint32_t size) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by dividing work across cores
  uint32_t chunk_size = size / numThreads;
  uint32_t remainder = size % numThreads;

  uint32_t start, end;
  if (core_id < remainder) {
    chunk_size += 1;
    start = core_id * chunk_size;
  } else {
    start = core_id * chunk_size + remainder;
  }
  end = start + chunk_size;

  // HardSwish(x) = x * clip(x/6 + 0.5, 0, 1)
  // Piecewise:
  //   x <= -3: output = 0
  //   -3 < x < 3: output = x * (x/6 + 0.5)
  //   x >= 3: output = x

  for (uint32_t i = start; i < end; i++) {
    float32_t x = data_in[i];
    float32_t clip_val = x / 6.0f + 0.5f;

    // Clamp to [0, 1]
    if (clip_val < 0.0f) {
      clip_val = 0.0f;
    } else if (clip_val > 1.0f) {
      clip_val = 1.0f;
    }

    data_out[i] = x * clip_val;
  }
}
