/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                              const float32_t *__restrict__ pSrcB,
                              const float32_t *__restrict__ pDstC,
                              float32_t *__restrict__ pDstY, uint32_t M,
                              uint32_t N, uint32_t O, int32_t transA,
                              int32_t transB) {
  for (uint32_t i = 0; i < M; ++i) {
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