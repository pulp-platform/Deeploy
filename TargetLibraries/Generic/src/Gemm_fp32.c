/* =====================================================================
 * Title:        GEMM_fp32.c
 * Description:
 *
 * Date:         24.01.2025
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
#include "DeeployBasicMath.h"

void Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                              const float32_t *__restrict__ pSrcB,
                              const float32_t *__restrict__ pDstC,
                              float32_t *__restrict__ pDstY, uint32_t M,
                              uint32_t N, uint32_t O, int32_t transA,
                              int32_t transB) {
  if (pDstC == NULL) {
    for (uint32_t i = 0; i < M; ++i) {
      for (uint32_t j = 0; j < O; ++j) {
        float32_t sum = 0.0f;
        for (uint32_t k = 0; k < N; ++k) {
          uint32_t a_idx = transA ? (k * M + i) : (i * N + k);
          uint32_t b_idx = transB ? (j * N + k) : (k * O + j);

          sum += pSrcA[a_idx] * pSrcB[b_idx];
        }
        pDstY[i * O + j] = sum;
      }
    }
  } else {
    for (uint32_t i = 0; i < M; ++i) {
      for (uint32_t j = 0; j < O; ++j) {
        float32_t sum = 0.0f;
        for (uint32_t k = 0; k < N; ++k) {
          uint32_t a_idx = transA ? (k * M + i) : (i * N + k);
          uint32_t b_idx = transB ? (j * N + k) : (k * O + j);

          sum += pSrcA[a_idx] * pSrcB[b_idx];
        }
        pDstY[i * O + j] = sum  + pDstC[i * O + j];
      }
    }
  }
}