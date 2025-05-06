/* =====================================================================
 * Title:        MatMul_s8.c
 * Description:
 *
 * $Date:        19.12.2022
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Moritz Scherer, ETH Zurich
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

void MatMul_s8_s8_s32(int8_t const *__restrict__ pSrcA,
                      int8_t const *__restrict__ pSrcB,
                      int32_t *__restrict__ pDstC, uint32_t M, uint32_t N,
                      uint32_t P, int32_t A_offset, int32_t B_offset,
                      int32_t C_offset) {
  uint32_t i = 0; // loop counter
  uint32_t j = 0; // loop counter
  uint32_t k = 0; // loop counter

  for (i = 0; i < M / 2; i++) {
    for (k = 0; k < P / 2; k++) {

      int32_t sum00 = C_offset;
      int32_t sum01 = C_offset;
      int32_t sum10 = C_offset;
      int32_t sum11 = C_offset;

      for (j = 0; j < N / 2; j++) {
        int32_t AVal00 = pSrcA[(i * 2) * N + j * 2] + A_offset;
        int32_t AVal10 = pSrcA[(i * 2 + 1) * N + j * 2] + A_offset;
        int32_t AVal01 = pSrcA[(i * 2) * N + j * 2 + 1] + A_offset;
        int32_t AVal11 = pSrcA[(i * 2 + 1) * N + j * 2 + 1] + A_offset;
        int32_t BVal00 = pSrcB[(j * 2) * P + (k * 2)] + B_offset;
        int32_t BVal01 = pSrcB[(j * 2) * P + (k * 2 + 1)] + B_offset;
        int32_t BVal10 = pSrcB[(j * 2 + 1) * P + (k * 2)] + B_offset;
        int32_t BVal11 = pSrcB[(j * 2 + 1) * P + (k * 2 + 1)] + B_offset;

        sum00 = sum00 + AVal00 * BVal00;
        sum00 = sum00 + AVal01 * BVal10;
        sum01 = sum01 + AVal00 * BVal01;
        sum01 = sum01 + AVal01 * BVal11;
        sum10 = sum10 + AVal10 * BVal00;
        sum10 = sum10 + AVal11 * BVal10;
        sum11 = sum11 + AVal10 * BVal01;
        sum11 = sum11 + AVal11 * BVal11;
      }
      pDstC[(i * 2) * P + (k * 2)] = sum00;
      pDstC[(i * 2) * P + (k * 2 + 1)] = sum01;
      pDstC[(i * 2 + 1) * P + (k * 2)] = sum10;
      pDstC[(i * 2 + 1) * P + (k * 2 + 1)] = sum11;
    }
  }

  // clean up code
  i = i * 2;
  j = j * 2;
  k = k * 2;

  // clean up code
  // check if every index is nicely finished
  if (i == M && j == N && k == P) {
    return;
  } else {
    uint32_t iEnd = i;
    uint32_t jEnd = j;
    uint32_t kEnd = k;

    // clean up for j
    if (jEnd != N) {
      for (i = 0; i < iEnd; i++) {
        for (k = 0; k < kEnd; k++) {
          int32_t sum = 0;
          for (j = jEnd; j < N; j++) {
            sum = sum +
                  (pSrcA[i * N + j] + A_offset) * (pSrcB[j * P + k] + B_offset);
          }
          pDstC[i * P + k] += sum;
        }
      }
    }

    // clean up for k
    if (kEnd != P) {
      for (i = 0; i < iEnd; i++) {
        for (k = kEnd; k < P; k++) {
          int32_t sum = C_offset;
          for (j = 0; j < N; j++) {
            sum = sum +
                  (pSrcA[i * N + j] + A_offset) * (pSrcB[j * P + k] + B_offset);
          }
          pDstC[i * P + k] = sum;
        }
      }
    }

    // clean up for i
    for (i = iEnd; i < M; i++) {
      for (k = 0; k < P; k++) {
        int32_t sum = C_offset;
        for (j = 0; j < N; j++) {
          sum = sum +
                (pSrcA[i * N + j] + A_offset) * (pSrcB[j * P + k] + B_offset);
        }
        pDstC[i * P + k] = sum;
      }
    }
  }
}
