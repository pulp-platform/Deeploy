/* =====================================================================
 * Title:        Gemm_s8.c
 * Description:
 *
 * $Date:        05.01.2023
 *
 * ===================================================================== */
/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
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

#include "DeeployBasicMath.h"

void Gemm_s8_s8_s32_s32(int8_t const *__restrict__ pSrcA,
                        int8_t const *__restrict__ pSrcB,
                        int32_t const *__restrict__ pSrcC,
                        int32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                        uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
                        int32_t transB, int32_t A_offset, int32_t B_offset,
                        int32_t C_offset, int32_t Y_offset) {

  uint32_t m = 0; // loop counter
  uint32_t n = 0; // loop counter
  uint32_t p = 0; // loop counter

  if (transA == 0 && transB == 0) {
    for (m = 0; m < M; ++m) {
      for (p = 0; p < P; p++) {
        int32_t sum = 0;
        for (n = 0; n < N; n++) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        pDstY[m * P + p] =
            alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
      }
    }
  } else if (transA == 1 && transB == 0) {
    for (uint32_t m = 0; m < M; ++m) {
      for (p = 0; p < P; p++) {
        int32_t sum = 0;
        for (n = 0; n < N; n++) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        pDstY[m * P + p] =
            alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
      }
    }
  } else if (transA == 0 && transB == 1) {
    for (uint32_t m = 0; m < M; ++m) {
      for (p = 0; p < P; p++) {
        int32_t sum = 0;
        for (n = 0; n < N; n++) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        pDstY[m * P + p] =
            alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
      }
    }
  } else {
    for (uint32_t m = 0; m < M; ++m) {
      for (p = 0; p < P; p++) {
        int32_t sum = 0;
        for (n = 0; n < N; n++) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        pDstY[m * P + p] =
            alpha * sum + beta * (pSrcC[m * P + p] + C_offset) + Y_offset;
      }
    }
  }
}
