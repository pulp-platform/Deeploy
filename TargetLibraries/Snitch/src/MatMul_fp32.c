/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

/*
 * Multi-core FP32 matrix multiplication (scalar, no SSR)
 *
 * Computes: Y = A * B
 * A is M x N, B is N x O, Y is M x O
 * All matrices in row-major layout.
 *
 * Splits M rows across compute cores internally.
 * Uses a distinct function name to avoid being shadowed by
 * the Generic single-core MatMul_fp32_fp32_fp32 (link order).
 */
void matmul_fp32_opt(const float32_t *__restrict__ pSrcA,
                     const float32_t *__restrict__ pSrcB,
                     float32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                     uint32_t O) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t rows_per_core = M / numThreads;
  uint32_t remainder = M % numThreads;

  uint32_t start_row, num_rows;
  if (core_id < remainder) {
    num_rows = rows_per_core + 1;
    start_row = core_id * num_rows;
  } else {
    num_rows = rows_per_core;
    start_row = core_id * rows_per_core + remainder;
  }

  const uint32_t unroll = 8;
  uint32_t O_block = O - (O % unroll);

  for (uint32_t i = start_row; i < start_row + num_rows; i++) {
    uint32_t j;
    for (j = 0; j < O_block; j += unroll) {
      float32_t c0 = 0.0f;
      float32_t c1 = 0.0f;
      float32_t c2 = 0.0f;
      float32_t c3 = 0.0f;
      float32_t c4 = 0.0f;
      float32_t c5 = 0.0f;
      float32_t c6 = 0.0f;
      float32_t c7 = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a = pSrcA[i * N + k];
        c0 += a * pSrcB[k * O + j + 0];
        c1 += a * pSrcB[k * O + j + 1];
        c2 += a * pSrcB[k * O + j + 2];
        c3 += a * pSrcB[k * O + j + 3];
        c4 += a * pSrcB[k * O + j + 4];
        c5 += a * pSrcB[k * O + j + 5];
        c6 += a * pSrcB[k * O + j + 6];
        c7 += a * pSrcB[k * O + j + 7];
      }

      pDstY[i * O + j + 0] = c0;
      pDstY[i * O + j + 1] = c1;
      pDstY[i * O + j + 2] = c2;
      pDstY[i * O + j + 3] = c3;
      pDstY[i * O + j + 4] = c4;
      pDstY[i * O + j + 5] = c5;
      pDstY[i * O + j + 6] = c6;
      pDstY[i * O + j + 7] = c7;
    }

    // Cleanup for remaining columns
    for (; j < O; j++) {
      float32_t sum = 0.0f;
      for (uint32_t k = 0; k < N; k++) {
        sum += pSrcA[i * N + k] * pSrcB[k * O + j];
      }
      pDstY[i * O + j] = sum;
    }
  }
}
