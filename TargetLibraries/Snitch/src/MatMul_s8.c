/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"
void MatMul_parallel_s8_rv32im(int8_t const *__restrict__ pSrcA,
                               int8_t const *__restrict__ pSrcB,
                               int32_t *__restrict__ pDstC, uint32_t M,
                               uint32_t N, uint32_t P, int32_t A_offset,
                               int32_t B_offset, int32_t output_offset) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);

  for (uint32_t i = core_id / c; i < M; i += numThreads / c) {
    for (uint32_t j = c_start; j < c_end; ++j) {
      int32_t sum = 0;
      for (uint32_t k = 0; k < N; ++k) {
        sum += (int32_t)(pSrcA[i * N + k] + A_offset) *
               (pSrcB[k * P + j] + B_offset);
      }
      pDstC[i * P + j] = sum + output_offset;
    }
  }
}

void MatMul_unrolled_2x2_parallel_s8_rv32im(int8_t const *__restrict__ pSrcA,
                                            int8_t const *__restrict__ pSrcB,
                                            int32_t *__restrict__ pDstC,
                                            uint32_t M, uint32_t N,
                                            uint32_t P) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);
  for (uint32_t i = 2 * (core_id / c); i < M; i += 2 * (numThreads / c)) {
    for (uint32_t j = c_start; j < c_end; j += 2) {
      int32_t c00 = 0;
      int32_t c01 = 0;
      int32_t c10 = 0;
      int32_t c11 = 0;
      for (uint32_t k = 0; k < N; k += 2) {
        // Explicitly load the values first to help with scheduling
        int8_t val_a00 = (int8_t)(pSrcA[(i + 0) * N + k + 0]);
        int8_t val_a01 = (int8_t)(pSrcA[(i + 0) * N + k + 1]);
        int8_t val_a10 = (int8_t)(pSrcA[(i + 1) * N + k + 0]);
        int8_t val_a11 = (int8_t)(pSrcA[(i + 1) * N + k + 1]);
        int8_t val_b00 = (int8_t)(pSrcB[(k + 0) * P + j + 0]);
        int8_t val_b01 = (int8_t)(pSrcB[(k + 0) * P + j + 1]);
        int8_t val_b10 = (int8_t)(pSrcB[(k + 1) * P + j + 0]);
        int8_t val_b11 = (int8_t)(pSrcB[(k + 1) * P + j + 1]);
        c00 += val_a00 * val_b00;
        c00 += val_a01 * val_b10;
        c01 += val_a00 * val_b01;
        c01 += val_a01 * val_b11;
        c10 += val_a10 * val_b00;
        c10 += val_a11 * val_b10;
        c11 += val_a10 * val_b01;
        c11 += val_a11 * val_b11;
      }
      pDstC[(i + 0) * P + j + 0] = c00;
      pDstC[(i + 0) * P + j + 1] = c01;
      pDstC[(i + 1) * P + j + 0] = c10;
      pDstC[(i + 1) * P + j + 1] = c11;
    }
  }
}

void MatMul_offset_unrolled_2x2_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    int32_t A_offset, int32_t B_offset, int32_t output_offset) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);
  for (uint32_t i = 2 * (core_id / c); i < M; i += 2 * (numThreads / c)) {
    for (uint32_t j = c_start; j < c_end; j += 2) {
      int32_t c00 = 0;
      int32_t c01 = 0;
      int32_t c10 = 0;
      int32_t c11 = 0;
      for (uint32_t k = 0; k < N; k += 2) {
        // Explicitly load the values first to help with scheduling
        int8_t val_a00 = (int8_t)(pSrcA[(i + 0) * N + k + 0] + A_offset);
        int8_t val_a01 = (int8_t)(pSrcA[(i + 0) * N + k + 1] + A_offset);
        int8_t val_a10 = (int8_t)(pSrcA[(i + 1) * N + k + 0] + A_offset);
        int8_t val_a11 = (int8_t)(pSrcA[(i + 1) * N + k + 1] + A_offset);
        int8_t val_b00 = (int8_t)(pSrcB[(k + 0) * P + j + 0] + B_offset);
        int8_t val_b01 = (int8_t)(pSrcB[(k + 0) * P + j + 1] + B_offset);
        int8_t val_b10 = (int8_t)(pSrcB[(k + 1) * P + j + 0] + B_offset);
        int8_t val_b11 = (int8_t)(pSrcB[(k + 1) * P + j + 1] + B_offset);
        c00 += val_a00 * val_b00;
        c00 += val_a01 * val_b10;
        c01 += val_a00 * val_b01;
        c01 += val_a01 * val_b11;
        c10 += val_a10 * val_b00;
        c10 += val_a11 * val_b10;
        c11 += val_a10 * val_b01;
        c11 += val_a11 * val_b11;
      }
      pDstC[(i + 0) * P + j + 0] = c00 + output_offset;
      pDstC[(i + 0) * P + j + 1] = c01 + output_offset;
      pDstC[(i + 1) * P + j + 0] = c10 + output_offset;
      pDstC[(i + 1) * P + j + 1] = c11 + output_offset;
    }
  }
}
