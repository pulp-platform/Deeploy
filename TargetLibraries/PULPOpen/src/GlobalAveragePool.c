/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"


void PULP_GlobalAveragePool_fp32(const float32_t *input, float32_t *output,
                                  uint32_t N, uint32_t C, uint32_t H, uint32_t W) {
  int8_t core_id  = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop  = MIN(ch_start + ch_chunk, C);

  uint32_t HW = H * W;
  float32_t inv_HW = 1.0f / (float32_t)HW;

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t c = ch_start; c < ch_stop; ++c) {
      float32_t sum = 0.0f;
      uint32_t in_base = (n * C + c) * HW;
      for (uint32_t i = 0; i < HW; ++i) {
        sum += input[in_base + i];
      }
      output[n * C + c] = sum * inv_HW;
    }
  }
}


void PULP_GlobalAveragePoolGrad_fp32(const float32_t *dY, float32_t *dX,
                                      uint32_t N, uint32_t C, uint32_t H, uint32_t W) {
  int8_t core_id  = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop  = MIN(ch_start + ch_chunk, C);

  uint32_t HW = H * W;
  float32_t inv_HW = 1.0f / (float32_t)HW;

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t c = ch_start; c < ch_stop; ++c) {
      float32_t dy_val = dY[n * C + c] * inv_HW;
      uint32_t out_base = (n * C + c) * HW;
      for (uint32_t i = 0; i < HW; ++i) {
        dX[out_base + i] = dy_val;
      }
    }
  }
}
