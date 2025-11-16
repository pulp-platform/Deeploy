/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployMath.h"
void Gemm_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int32_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, uint32_t core_id, uint32_t numThreads) {

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);

  const int32_t bias = beta * C_offset + Y_offset;

  if (transA == 0 && transB == 0) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        pDstY[m * P + p] = alpha * sum + beta * pSrcC[m * P + p] + bias;
      }
    }
  } else if (transA == 1 && transB == 0) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        pDstY[m * P + p] = alpha * sum + beta * pSrcC[m * P + p] + bias;
      }
    }
  } else if (transA == 0 && transB == 1) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        pDstY[m * P + p] = alpha * sum + beta * pSrcC[m * P + p] + bias;
      }
    }
  } else {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        pDstY[m * P + p] = alpha * sum + beta * pSrcC[m * P + p] + bias;
      }
    }
  }
}
