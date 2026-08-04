/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

/*
 * Element-wise Multiplication (FP32) with optional scalar broadcasting.
 *
 * is_scalar == 0:  output[i] = input1[i] * input2[i]
 * is_scalar != 0:  output[i] = input1[i] * input2[0]   (input2 read as scalar)
 *
 * input1:    First input tensor (float32)
 * input2:    Second input tensor (float32). Only input2[0] is read when
 *            is_scalar != 0.
 * output:    Output tensor (same shape as input1)
 * size:      Total number of elements in input1 / output
 * is_scalar: Non-zero selects the scalar-broadcast branch.
 *
 * multi-core      = yes
 * parallelization = element-wise across input1
 */
void Mul_fp32(float32_t *input1, float32_t *input2, float32_t *output,
              uint32_t size, uint32_t is_scalar) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t elements_per_core = size / numThreads;
  uint32_t remainder = size % numThreads;

  uint32_t start_elem, num_elems;
  if (core_id < remainder) {
    num_elems = elements_per_core + 1;
    start_elem = core_id * num_elems;
  } else {
    num_elems = elements_per_core;
    start_elem = core_id * elements_per_core + remainder;
  }

  if (is_scalar) {
    float32_t scalar = input2[0];
    for (uint32_t i = start_elem; i < start_elem + num_elems; i++) {
      output[i] = input1[i] * scalar;
    }
  } else {
    for (uint32_t i = start_elem; i < start_elem + num_elems; i++) {
      output[i] = input1[i] * input2[i];
    }
  }
}
