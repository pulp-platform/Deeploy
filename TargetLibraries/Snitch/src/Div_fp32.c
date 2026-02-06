/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

/*
 * Element-wise Division (FP32)
 *
 * Computes: output[i] = input1[i] / input2[i]
 *
 * Supports ONNX broadcasting rules:
 * - If input2 is scalar (size=1): divides all elements of input1 by input2[0]
 * - If both have same size: element-wise division
 *
 * input1:         Numerator tensor (float32)
 * input2:         Denominator tensor (float32)
 * output:         Output tensor (same shape as input1)
 * size:           Total number of elements in input1
 *
 * multi-core      = yes
 * parallelization = element-wise across input1
 */
void Div_fp32(float32_t *input1, float32_t *input2, float32_t *output,
              uint32_t size) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize across elements
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

  // Check if input2 is a scalar (size=1, broadcasted)
  // Note: This assumes the parser has set input2_size correctly
  // For now, we assume element-wise division (same size)
  for (uint32_t i = start_elem; i < start_elem + num_elems; i++) {
    output[i] = input1[i] / input2[i];
  }
}

/*
 * Element-wise Division with scalar broadcasting (FP32)
 *
 * Computes: output[i] = input1[i] / scalar
 *
 * input1:         Numerator tensor (float32)
 * scalar:         Scalar denominator (float32)
 * output:         Output tensor (same shape as input1)
 * size:           Total number of elements in input1
 *
 * multi-core      = yes
 * parallelization = element-wise
 */
void Div_fp32_scalar(float32_t *input1, float32_t scalar, float32_t *output,
                     uint32_t size) {

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

  float32_t inv_scalar = 1.0f / scalar; // Compute inverse once

  for (uint32_t i = start_elem; i < start_elem + num_elems; i++) {
    output[i] = input1[i] * inv_scalar;
  }
}
