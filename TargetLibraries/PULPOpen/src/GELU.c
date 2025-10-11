/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

#define M_PI 3.14159265358979323846

void PULP_GELU_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         int32_t dataSize) {
  // Get core information
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  // Split into chunks for each core
  int16_t chunk = (dataSize >> log2Core) + ((dataSize & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, dataSize);
  int16_t chunk_stop = MIN(chunk_start + chunk, dataSize);

  // Compute GELU on the assigned chunk
  for (uint32_t i = chunk_start; i < chunk_stop; i++) {
    float32_t x = data_in[i];
    float32_t cdf = 0.5f * (1.0f + tanhf((sqrtf(2.0f / (float)M_PI) *
                                          (x + 0.044715f * powf(x, 3.0f)))));

    data_out[i] = x * cdf;
  }
}

void PULP_GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out,
                                 int32_t dataSize) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);
  int16_t chunk = (dataSize >> log2Core) + ((dataSize & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, dataSize);
  int16_t chunk_stop = MIN(chunk_start + chunk, dataSize);

  const float32_t scale = 1.702f;
  for (uint32_t i = chunk_start; i < chunk_stop; i++) {
    float32_t x = data_in[i];
    float32_t sigmoid_in = scale * x;
    // sigmoid(z) = 1 / (1 + exp(-z))
    float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
    data_out[i] = x * sigmoid;
  }
}