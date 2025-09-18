/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployMath.h"

void ITAMax_parallel_s8(int8_t const *__restrict__ pSrcA,
                        int8_t *__restrict__ pDstB, int8_t *__restrict__ pBufN,
                        uint32_t size, uint32_t lastDimLength,
                        uint32_t n_levels, uint32_t core_id,
                        uint32_t numThreads) {

  uint32_t i = 0; // Row Counter
  uint32_t j = 0; // Column Counter

  uint8_t *shift = (uint8_t *)pBufN;

  for (i = core_id; i < size / lastDimLength; i += numThreads) {
    // 1. Find maximum over row
    int8_t max = -128;
    for (j = 0; j < lastDimLength; ++j) {
      if (pSrcA[i * lastDimLength + j] > max) {
        max = pSrcA[i * lastDimLength + j];
      }
    }

    // 2. Calculate exponential sum
    uint32_t exp_sum = 0;
    for (j = 0; j < lastDimLength; ++j) {
      int32_t diff = max - pSrcA[i * lastDimLength + j];
      shift[j + lastDimLength * core_id] = (uint8_t)((diff + 16) >> 5);
      exp_sum += (256U >> shift[j + lastDimLength * core_id]);
    }

    uint32_t exp_sum_inv = ((n_levels - 1) * 256U) / exp_sum;

    for (j = 0; j < lastDimLength; ++j) {
      pDstB[i * lastDimLength + j] =
          (int8_t)((exp_sum_inv >> shift[j + lastDimLength * core_id]) -
                   (n_levels / 2));
    }
  }
}
