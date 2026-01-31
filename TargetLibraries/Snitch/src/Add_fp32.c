/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

void Add_fp32(float32_t *pIn1, float32_t *pIn2, float32_t *pOut,
              uint32_t size) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t chunkSize = size / numThreads;
  uint32_t remainder = size % numThreads;

  uint32_t start, num_elements;
  if (core_id < remainder) {
    num_elements = chunkSize + 1;
    start = core_id * num_elements;
  } else {
    num_elements = chunkSize;
    start = core_id * chunkSize + remainder;
  }

  uint32_t end = start + num_elements;

  for (uint32_t i = start; i < end; i++) {
    pOut[i] = pIn1[i] + pIn2[i];
  }
}

void Add_fp32_broadcast(float32_t *pIn1, float32_t *pIn2, float32_t *pOut,
                        uint32_t *out_shape, uint32_t *strides1,
                        uint32_t *strides2, uint32_t ndim, uint32_t size) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t chunkSize = size / numThreads;
  uint32_t remainder = size % numThreads;

  uint32_t start, num_elements;
  if (core_id < remainder) {
    num_elements = chunkSize + 1;
    start = core_id * num_elements;
  } else {
    num_elements = chunkSize;
    start = core_id * chunkSize + remainder;
  }

  uint32_t end = start + num_elements;

  for (uint32_t i = start; i < end; i++) {
    uint32_t idx1 = 0;
    uint32_t idx2 = 0;
    uint32_t tmp = i;

    for (int32_t d = ndim - 1; d >= 0; d--) {
      uint32_t coord = tmp % out_shape[d];
      tmp /= out_shape[d];
      idx1 += coord * strides1[d];
      idx2 += coord * strides2[d];
    }

    pOut[i] = pIn1[idx1] + pIn2[idx2];
  }
}

void Add_fp32_lastdim(float32_t *pIn1, float32_t *pIn2, float32_t *pOut,
                      uint32_t outer_size, uint32_t inner_size) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();
  uint32_t size = outer_size * inner_size;

  uint32_t chunkSize = size / numThreads;
  uint32_t remainder = size % numThreads;

  uint32_t start, num_elements;
  if (core_id < remainder) {
    num_elements = chunkSize + 1;
    start = core_id * num_elements;
  } else {
    num_elements = chunkSize;
    start = core_id * chunkSize + remainder;
  }

  uint32_t end = start + num_elements;

  for (uint32_t i = start; i < end; i++) {
    uint32_t inner_idx = i % inner_size;
    pOut[i] = pIn1[i] + pIn2[inner_idx];
  }
}
