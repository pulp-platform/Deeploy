/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_Relu_fp32_fp32(float32_t *input, float32_t *output, uint32_t size) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t start = MIN(chunk * core_id, size);
  int32_t end = MIN(start + chunk, size);
  int32_t local_size = end - start;

  float32_t *local_input = input + start;
  float32_t *local_output = output + start;

  for (int32_t i = 0; i < local_size; i++) {
    local_output[i] = MAX(local_input[i], 0.0f);
  }
}

void PULP_ReluGrad_fp32_fp32(float32_t *grad_out, float32_t *data_in,
                             float32_t *grad_in, uint32_t size) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  int32_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int32_t start = MIN(chunk * core_id, size);
  int32_t end = MIN(start + chunk, size);
  int32_t local_size = end - start;

  float32_t *local_grad_out = grad_out + start;
  float32_t *local_data_in = data_in + start;
  float32_t *local_grad_in = grad_in + start;

  for (int32_t i = 0; i < local_size; i++) {
    // If input > 0, gradient flows through; otherwise gradient is 0
    local_grad_in[i] = (local_data_in[i] > 0.0f) ? local_grad_out[i] : 0.0f;
  }
}