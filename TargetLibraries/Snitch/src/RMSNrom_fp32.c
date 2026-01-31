/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"
#include <math.h>

void RMSNorm_fp32(float32_t *data_in, float32_t *weight, float32_t *data_out,
                  uint32_t size, uint32_t lastDimLength, float32_t eps) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t num_vectors = size / lastDimLength;

  // Parallelize across vectors (batch * sequence dimension)
  uint32_t vectors_per_core = num_vectors / numThreads;
  uint32_t remainder = num_vectors % numThreads;

  uint32_t start_vec, num_vecs;
  if (core_id < remainder) {
    num_vecs = vectors_per_core + 1;
    start_vec = core_id * num_vecs;
  } else {
    num_vecs = vectors_per_core;
    start_vec = core_id * vectors_per_core + remainder;
  }

  for (uint32_t v = start_vec; v < start_vec + num_vecs; v++) {
    float32_t *in_ptr = data_in + v * lastDimLength;
    float32_t *out_ptr = data_out + v * lastDimLength;

    // Compute sum of squares
    float32_t sum_sq = 0.0f;
    for (uint32_t i = 0; i < lastDimLength; i++) {
      sum_sq += in_ptr[i] * in_ptr[i];
    }

    // Compute RMS with epsilon
    float32_t rms = sqrtf(sum_sq / (float32_t)lastDimLength + eps);
    float32_t inv_rms = 1.0f / rms;

    // Apply normalization and weight
    for (uint32_t i = 0; i < lastDimLength; i++) {
      out_ptr[i] = in_ptr[i] * inv_rms * weight[i];
    }
  }
}
