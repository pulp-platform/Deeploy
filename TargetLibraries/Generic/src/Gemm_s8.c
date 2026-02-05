/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
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
