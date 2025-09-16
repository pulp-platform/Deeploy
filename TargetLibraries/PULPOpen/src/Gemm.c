
/* =====================================================================
 * Title:        Gemm.c
 * Description:
 *
 * Date:         05.06.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Run Wang, ETH Zurich
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

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                                   const float32_t *__restrict__ pSrcB,
                                   const float32_t *__restrict__ pDstC,
                                   float32_t *__restrict__ pDstY, uint32_t M,
                                   uint32_t N, uint32_t O, uint32_t transA,
                                   uint32_t transB) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN(core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0) {
    return;
  }

  for (uint32_t i = M_start; i < M_end; ++i) {
    for (uint32_t j = 0; j < O; ++j) {
      float32_t sum = 0.0f;
      for (uint32_t k = 0; k < N; ++k) {
        uint32_t a_idx = transA ? (k * M + i) : (i * N + k);
        uint32_t b_idx = transB ? (j * N + k) : (k * O + j);
        sum += pSrcA[a_idx] * pSrcB[b_idx];
      }
      pDstY[i * O + j] = sum + pDstC[i * O + j];
    }
  }
}