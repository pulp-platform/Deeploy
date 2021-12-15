/* =====================================================================
 * Title:        Softmax_s8.c
 * Description:
 *
 * $Date:        25.04.2023
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
