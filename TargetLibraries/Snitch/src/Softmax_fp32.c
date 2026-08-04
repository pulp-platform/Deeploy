/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"
#include <math.h>

/*
 * Multi-core FP32 Softmax
 *
 * Computes softmax along the last dimension:
 *   output[b][i] = exp(input[b][i] - max) / sum(exp(input[b][j] - max))
 *
 * Parallelizes across the batch dimension (size / lastDimLength rows).
 *
 * input:          Input tensor (float32)
 * output:         Output tensor (float32)
 * size:           Total number of elements
 * lastDimLength:  Length of the last dimension (softmax axis)
 */
void Softmax_fp32(float32_t *input, float32_t *output, uint32_t size,
                  uint32_t lastDimLength) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t num_rows = size / lastDimLength;

  uint32_t rows_per_core = num_rows / numThreads;
  uint32_t remainder = num_rows % numThreads;

  uint32_t start_row, num_rows_this_core;
  if (core_id < remainder) {
    num_rows_this_core = rows_per_core + 1;
    start_row = core_id * num_rows_this_core;
  } else {
    num_rows_this_core = rows_per_core;
    start_row = core_id * rows_per_core + remainder;
  }

  for (uint32_t r = start_row; r < start_row + num_rows_this_core; r++) {
    float32_t *in_row = input + r * lastDimLength;
    float32_t *out_row = output + r * lastDimLength;

    // Find max for numerical stability
    float32_t max_val = -INFINITY;
    for (uint32_t i = 0; i < lastDimLength; i++) {
      if (in_row[i] > max_val)
        max_val = in_row[i];
    }

    // Compute exp and sum
    float32_t sum = 0.0f;
    for (uint32_t i = 0; i < lastDimLength; i++) {
      out_row[i] = expf(in_row[i] - max_val);
      sum += out_row[i];
    }

    // Normalize
    float32_t inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < lastDimLength; i++) {
      out_row[i] *= inv_sum;
    }
  }
}
