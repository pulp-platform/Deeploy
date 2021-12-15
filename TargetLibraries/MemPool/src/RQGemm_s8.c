/* =====================================================================
 * Title:        RQGemm_s8.c
 * Description:
 *
 * Date:         16.05.2023
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

#include "DeeployMath.h"
void RQGemm_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, int8_t output_min, int8_t output_max, uint32_t core_id,
    uint32_t numThreads) {

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);

  const int32_t rqs_bias = ((1 << (log2D - 1))) * rounding;
  const int32_t bias = beta * C_offset;

  int32_t _add = add[0];
  int32_t _mul = mul[0];

  if (transA == 0 && transB == 0) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  } else if (transA == 1 && transB == 0) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  } else if (transA == 0 && transB == 1) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  } else {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  }
}

void RQGemm_offset_unrolled_2x2_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, uint32_t core_id, uint32_t numThreads) {

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);

  const int32_t rqs_bias = ((1 << (log2D - 1))) * rounding;
  const int32_t bias = beta * C_offset;

  int32_t _add0 = add[0];
  int32_t _add1 = add[0];
  int32_t _mul0 = mul[0];
  int32_t _mul1 = mul[0];

  if (transA == 0 && transB == 0) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(m + 0) * N + n + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(m + 0) * N + n + 1] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(m + 1) * N + n + 0] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(m + 1) * N + n + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(n + 0) * P + p + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(n + 0) * P + p + 1] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(n + 1) * P + p + 0] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(n + 1) * P + p + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  } else if (transA == 1 && transB == 0) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(n + 0) * M + m + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(n + 1) * M + m + 0] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(n + 0) * M + m + 1] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(n + 1) * M + m + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(n + 0) * P + p + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(n + 0) * P + p + 1] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(n + 1) * P + p + 0] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(n + 1) * P + p + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  } else if (transA == 0 && transB == 1) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(m + 0) * N + n + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(m + 0) * N + n + 1] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(m + 1) * N + n + 0] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(m + 1) * N + n + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(p + 0) * N + n + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(p + 1) * N + n + 0] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(p + 0) * N + n + 1] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(p + 1) * N + n + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  } else if (transA == 1 && transB == 1) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(n + 0) * M + m + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(n + 1) * M + m + 0] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(n + 0) * M + m + 1] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(n + 1) * M + m + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(p + 0) * N + n + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(p + 1) * N + n + 0] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(p + 0) * N + n + 1] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(p + 1) * N + n + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  }
}

#ifdef __XPULPIMG
void RQGemm_offset_unrolled_4x4_pincr_asm_parallel_s8_xpulpv2(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, uint32_t core_id, uint32_t numThreads) {

  const int32_t rqs_bias = ((1 << (log2D - 1))) * rounding;
  const int32_t bias = beta * C_offset;

  const int32_t *idx_add = &add[0];
  const int32_t *idx_mul = &mul[0];
  // Loop counter for P
  uint32_t k = 0;
  if (transA == 0 && transB == 0) {
    // Masks for shuffles
    static v4s mask0 = {0, 1, 4, 5};
    static v4s mask1 = {2, 3, 6, 7};
    static v4s mask2 = {0, 2, 4, 6};
    static v4s mask3 = {1, 3, 5, 7};

    // Row decrement for A matrix
    int32_t const N_decr = -(int)N + 4;
    // Row increment for C matrix
    uint32_t const P_incr = (P * 4) - 12;

    v4s aVecOffset = {(int8_t)A_offset, (int8_t)A_offset, (int8_t)A_offset,
                      (int8_t)A_offset};
    v4s bVecOffset = {(int8_t)B_offset, (int8_t)B_offset, (int8_t)B_offset,
                      (int8_t)B_offset};

    for (k = core_id; k < P / 4; k += numThreads) {
      const int8_t *idx_a = &pSrcA[0];      // start_a
      const int32_t *idx_c = &pSrcC[k * 4]; // start_c
      int8_t *idx_y = &pDstY[k * 4];        // start_y
      int8_t const *end_y = &pDstY[P * M];  // actually (P * M) + (k * 4)
      while (idx_y < end_y) {
        int32_t sum00 = 0;
        int32_t sum01 = 0;
        int32_t sum02 = 0;
        int32_t sum03 = 0;
        int32_t sum10 = 0;
        int32_t sum11 = 0;
        int32_t sum12 = 0;
        int32_t sum13 = 0;

        v4s sum0, sum1;

        int8_t const *end_a = idx_a + N;
        const int8_t *idx_b = &pSrcB[k * 4]; // start_b
        while (idx_a < end_a) {
          v4s aVec0, aVec1;

          v4s temp0, temp1, temp2, temp3;

          __asm__ volatile(
              "p.lw %[a0], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[a1], %[a_decr](%[addr_a]!) \n\t"
              "p.lw %[t0], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[t1], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[t2], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[t3], %[b_incr](%[addr_b]!) \n\t"
              : [a0] "=&r"(aVec0), [a1] "=&r"(aVec1), [t0] "=&r"(temp0),
                [t1] "=&r"(temp1), [t2] "=&r"(temp2), [t3] "=&r"(temp3),
                [addr_a] "+&r"(idx_a), [addr_b] "+&r"(idx_b)
              : [a_incr] "r"(N), [a_decr] "r"(N_decr), [b_incr] "r"(P)
              : "memory");
          /* The asm code above implements the following commented C code */
          // go to next row, same column
          // v4s aVec0 = *((v4s *)idx_a); idx_a += N;
          // go to previous row, one column forward
          // v4s aVec1 = *((v4s *)idx_a); idx_a -= N - 4;
          // v4s temp0 = *((v4s *)idx_b); idx_b += P;
          // v4s temp1 = *((v4s *)idx_b); idx_b += P;
          // v4s temp2 = *((v4s *)idx_b); idx_b += P;
          // v4s temp3 = *((v4s *)idx_b); idx_b += P;
          aVec0 = __ADD4(aVec0, aVecOffset);
          aVec1 = __ADD4(aVec1, aVecOffset);

          // Shuffles to transpose at runtime the chunk extracted from B before
          // multiplying with A chunk temp0-3 variables needed because shuffles
          // use rD as source, but also modify it, thus we need a copy of their
          // content to use it twice in their original form
          v4s temp4 = __builtin_shuffle(temp0, temp1, mask0); // 0,1,4,5
          v4s temp5 = __builtin_shuffle(temp2, temp3, mask0); // 8,9,12,13
          v4s temp6 = __builtin_shuffle(temp0, temp1, mask1); // 2,3,6,7
          v4s temp7 = __builtin_shuffle(temp2, temp3, mask1); // 3,7,11,15

          v4s bVec0 = __builtin_shuffle(temp4, temp5, mask2); // 0,4,8,12
          v4s bVec1 = __builtin_shuffle(temp4, temp5, mask3); // 1,5,9,13
          v4s bVec2 = __builtin_shuffle(temp6, temp7, mask2); // 2,6,10,14
          v4s bVec3 = __builtin_shuffle(temp6, temp7, mask3); // 3,7,11,15

          bVec0 = __ADD4(bVec0, bVecOffset);
          bVec1 = __ADD4(bVec1, bVecOffset);
          bVec2 = __ADD4(bVec2, bVecOffset);
          bVec3 = __ADD4(bVec3, bVecOffset);

          sum00 = __SUMDOTP4(aVec0, bVec0, sum00);
          sum01 = __SUMDOTP4(aVec0, bVec1, sum01);
          sum02 = __SUMDOTP4(aVec0, bVec2, sum02);
          sum03 = __SUMDOTP4(aVec0, bVec3, sum03);
          sum10 = __SUMDOTP4(aVec1, bVec0, sum10);
          sum11 = __SUMDOTP4(aVec1, bVec1, sum11);
          sum12 = __SUMDOTP4(aVec1, bVec2, sum12);
          sum13 = __SUMDOTP4(aVec1, bVec3, sum13);
        }
        int32_t bias00, bias01, bias02, bias03;
        int32_t bias10, bias11, bias12, bias13;

        __asm__ volatile(
            "p.lw %[b00], 4(%[addr_c]!) \n\t"
            "p.lw %[b01], 4(%[addr_c]!) \n\t"
            "p.lw %[b02], 4(%[addr_c]!) \n\t"
            "p.lw %[b03], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b10], 4(%[addr_c]!) \n\t"
            "p.lw %[b11], 4(%[addr_c]!) \n\t"
            "p.lw %[b12], 4(%[addr_c]!) \n\t"
            "p.lw %[b13], %[c_incr](%[addr_c]!) \n\t"
            : [b00] "=&r"(bias00), [b01] "=&r"(bias01), [b02] "=&r"(bias02),
              [b03] "=&r"(bias03), [b10] "=&r"(bias10), [b11] "=&r"(bias11),
              [b12] "=&r"(bias12), [b13] "=&r"(bias13), [addr_c] "+&r"(idx_c)
            : [c_incr] "r"(P_incr)
            : "memory");

        sum00 = alpha * sum00 + beta * bias00 + bias;
        sum01 = alpha * sum01 + beta * bias01 + bias;
        sum02 = alpha * sum02 + beta * bias02 + bias;
        sum03 = alpha * sum03 + beta * bias03 + bias;
        sum10 = alpha * sum10 + beta * bias10 + bias;
        sum11 = alpha * sum11 + beta * bias11 + bias;
        sum12 = alpha * sum12 + beta * bias12 + bias;
        sum13 = alpha * sum13 + beta * bias13 + bias;

        int32_t _add0, _add1;
        int32_t _mul0, _mul1;
        if (per_row_quant) {
          __asm__ volatile(
              "p.lw %[add0], 4(%[addr_add]!) \n\t"
              "p.lw %[add1], 4(%[addr_add]!) \n\t"
              "p.lw %[mul0], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul1], 4(%[addr_mul]!) \n\t"
              : [add0] "=&r"(_add0), [mul0] "=&r"(_mul0), [add1] "=&r"(_add1),
                [mul1] "=&r"(_mul1), [addr_add] "+&r"(idx_add),
                [addr_mul] "+&r"(idx_mul)::"memory");
        } else {
          _add0 = add[0];
          _add1 = add[0];
          _mul0 = mul[0];
          _mul1 = mul[0];
        }

        sum00 = sum00 * _mul0 + rqs_bias + _add0;
        sum01 = sum01 * _mul0 + rqs_bias + _add0;
        sum02 = sum02 * _mul0 + rqs_bias + _add0;
        sum03 = sum03 * _mul0 + rqs_bias + _add0;
        sum10 = sum10 * _mul1 + rqs_bias + _add1;
        sum11 = sum11 * _mul1 + rqs_bias + _add1;
        sum12 = sum12 * _mul1 + rqs_bias + _add1;
        sum13 = sum13 * _mul1 + rqs_bias + _add1;

        sum00 = (sum00 >> log2D) + Y_offset;
        sum01 = (sum01 >> log2D) + Y_offset;
        sum02 = (sum02 >> log2D) + Y_offset;
        sum03 = (sum03 >> log2D) + Y_offset;
        sum10 = (sum10 >> log2D) + Y_offset;
        sum11 = (sum11 >> log2D) + Y_offset;
        sum12 = (sum12 >> log2D) + Y_offset;
        sum13 = (sum13 >> log2D) + Y_offset;

        sum0[0] = (int8_t)__CLIP(sum00, 7);
        sum0[1] = (int8_t)__CLIP(sum01, 7);
        sum0[2] = (int8_t)__CLIP(sum02, 7);
        sum0[3] = (int8_t)__CLIP(sum03, 7);
        sum1[0] = (int8_t)__CLIP(sum10, 7);
        sum1[1] = (int8_t)__CLIP(sum11, 7);
        sum1[2] = (int8_t)__CLIP(sum12, 7);
        sum1[3] = (int8_t)__CLIP(sum13, 7);

        __asm__ volatile("p.sw %[s0], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s1], %[y_incr](%[addr_y]!) \n\t"
                         : [addr_y] "+&r"(idx_y)
                         : [s0] "r"(sum0), [s1] "r"(sum1), [y_incr] "r"(P)
                         : "memory");
        /* The asm code above implements the following commented C code */
        // *(idx_y) = sum0; idx_y += P;
        // *(idx_y) = sum1; idx_y += P;

        idx_a += N; // adjust A matrix pointer
      }
    }
  } else if (transA == 1 && transB == 0) {
    // Masks for shuffles
    static v4s mask0 = {0, 1, 4, 5};
    static v4s mask1 = {2, 3, 6, 7};
    static v4s mask2 = {0, 2, 4, 6};
    static v4s mask3 = {1, 3, 5, 7};

    // Row increment for C matrix
    uint32_t const P_incr = (P * 4) - 12;

    v4s aVecOffset = {(int8_t)A_offset, (int8_t)A_offset, (int8_t)A_offset,
                      (int8_t)A_offset};
    v4s bVecOffset = {(int8_t)B_offset, (int8_t)B_offset, (int8_t)B_offset,
                      (int8_t)B_offset};

    for (k = core_id; k < P / 4; k += numThreads) {
      const int8_t *idx_a = &pSrcA[0];      // start_a
      const int32_t *idx_c = &pSrcC[k * 4]; // start_c
      int8_t *idx_y = &pDstY[k * 4];        // start_y
      int8_t const *end_y = &pDstY[P * M];  // actually (P * M) + (k * 4)
      while (idx_y < end_y) {
        int32_t sum00 = 0;
        int32_t sum01 = 0;
        int32_t sum02 = 0;
        int32_t sum03 = 0;
        int32_t sum10 = 0;
        int32_t sum11 = 0;
        int32_t sum12 = 0;
        int32_t sum13 = 0;
        int32_t sum20 = 0;
        int32_t sum21 = 0;
        int32_t sum22 = 0;
        int32_t sum23 = 0;
        int32_t sum30 = 0;
        int32_t sum31 = 0;
        int32_t sum32 = 0;
        int32_t sum33 = 0;

        v4s sum0, sum1, sum2, sum3;

        int8_t const *end_a = idx_a + N * M;
        const int8_t *idx_b = &pSrcB[k * 4]; // start_b
        while (idx_a < end_a) {
          v4s bTemp0, bTemp1, bTemp2, bTemp3;
          v4s aTemp0, aTemp1, aTemp2, aTemp3;

          __asm__ volatile(
              "p.lw %[at0], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[at1], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[at2], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[at3], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[bt0], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[bt1], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[bt2], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[bt3], %[b_incr](%[addr_b]!) \n\t"
              : [at0] "=&r"(aTemp0), [at1] "=&r"(aTemp1), [at2] "=&r"(aTemp2),
                [at3] "=&r"(aTemp3), [bt0] "=&r"(bTemp0), [bt1] "=&r"(bTemp1),
                [bt2] "=&r"(bTemp2), [bt3] "=&r"(bTemp3), [addr_a] "+&r"(idx_a),
                [addr_b] "+&r"(idx_b)
              : [a_incr] "r"(M), [b_incr] "r"(P)
              : "memory");
          /* The asm code above implements the following commented C code */
          // go to next row, same column
          // v4s aTemp0 = *((v4s *)idx_a); idx_a += M;
          // v4s aTemp1 = *((v4s *)idx_a); idx_a += M;
          // v4s aTemp2 = *((v4s *)idx_a); idx_a += M;
          // v4s aTemp3 = *((v4s *)idx_a); idx_a += M;

          // v4s bTemp0 = *((v4s *)idx_b); idx_b += P;
          // v4s bTemp1 = *((v4s *)idx_b); idx_b += P;
          // v4s bTemp2 = *((v4s *)idx_b); idx_b += P;
          // v4s bTemp3 = *((v4s *)idx_b); idx_b += P;

          // Shuffles to transpose at runtime the chunk extracted from B before
          // multiplying with A
          v4s bTemp4 = __builtin_shuffle(bTemp0, bTemp1, mask0); // 0,1,4,5
          v4s bTemp5 = __builtin_shuffle(bTemp2, bTemp3, mask0); // 8,9,12,13
          v4s bTemp6 = __builtin_shuffle(bTemp0, bTemp1, mask1); // 2,3,6,7
          v4s bTemp7 = __builtin_shuffle(bTemp2, bTemp3, mask1); // 3,7,11,15

          v4s bVec0 = __builtin_shuffle(bTemp4, bTemp5, mask2); // 0,4,8,12
          v4s bVec1 = __builtin_shuffle(bTemp4, bTemp5, mask3); // 1,5,9,13
          v4s bVec2 = __builtin_shuffle(bTemp6, bTemp7, mask2); // 2,6,10,14
          v4s bVec3 = __builtin_shuffle(bTemp6, bTemp7, mask3); // 3,7,11,15

          // Shuffles to transpose at runtime the chunk extracted from A before
          // multiplying with B
          v4s aTemp4 = __builtin_shuffle(aTemp0, aTemp1, mask0); // 0,1,4,5
          v4s aTemp5 = __builtin_shuffle(aTemp2, aTemp3, mask0); // 8,9,12,13
          v4s aTemp6 = __builtin_shuffle(aTemp0, aTemp1, mask1); // 2,3,6,7
          v4s aTemp7 = __builtin_shuffle(aTemp2, aTemp3, mask1); // 3,7,11,15

          v4s aVec0 = __builtin_shuffle(aTemp4, aTemp5, mask2); // 0,4,8,12
          v4s aVec1 = __builtin_shuffle(aTemp4, aTemp5, mask3); // 1,5,9,13
          v4s aVec2 = __builtin_shuffle(aTemp6, aTemp7, mask2); // 2,6,10,14
          v4s aVec3 = __builtin_shuffle(aTemp6, aTemp7, mask3); // 3,7,11,15

          aVec0 = __ADD4(aVec0, aVecOffset);
          aVec1 = __ADD4(aVec1, aVecOffset);
          aVec2 = __ADD4(aVec2, aVecOffset);
          aVec3 = __ADD4(aVec3, aVecOffset);

          bVec0 = __ADD4(bVec0, bVecOffset);
          bVec1 = __ADD4(bVec1, bVecOffset);
          bVec2 = __ADD4(bVec2, bVecOffset);
          bVec3 = __ADD4(bVec3, bVecOffset);

          sum00 = __SUMDOTP4(aVec0, bVec0, sum00);
          sum01 = __SUMDOTP4(aVec0, bVec1, sum01);
          sum02 = __SUMDOTP4(aVec0, bVec2, sum02);
          sum03 = __SUMDOTP4(aVec0, bVec3, sum03);
          sum10 = __SUMDOTP4(aVec1, bVec0, sum10);
          sum11 = __SUMDOTP4(aVec1, bVec1, sum11);
          sum12 = __SUMDOTP4(aVec1, bVec2, sum12);
          sum13 = __SUMDOTP4(aVec1, bVec3, sum13);
          sum20 = __SUMDOTP4(aVec2, bVec0, sum20);
          sum21 = __SUMDOTP4(aVec2, bVec1, sum21);
          sum22 = __SUMDOTP4(aVec2, bVec2, sum22);
          sum23 = __SUMDOTP4(aVec2, bVec3, sum23);
          sum30 = __SUMDOTP4(aVec3, bVec0, sum30);
          sum31 = __SUMDOTP4(aVec3, bVec1, sum31);
          sum32 = __SUMDOTP4(aVec3, bVec2, sum32);
          sum33 = __SUMDOTP4(aVec3, bVec3, sum33);
        }
        int32_t bias00, bias01, bias02, bias03;
        int32_t bias10, bias11, bias12, bias13;
        int32_t bias20, bias21, bias22, bias23;
        int32_t bias30, bias31, bias32, bias33;

        __asm__ volatile(
            "p.lw %[b00], 4(%[addr_c]!) \n\t"
            "p.lw %[b01], 4(%[addr_c]!) \n\t"
            "p.lw %[b02], 4(%[addr_c]!) \n\t"
            "p.lw %[b03], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b10], 4(%[addr_c]!) \n\t"
            "p.lw %[b11], 4(%[addr_c]!) \n\t"
            "p.lw %[b12], 4(%[addr_c]!) \n\t"
            "p.lw %[b13], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b20], 4(%[addr_c]!) \n\t"
            "p.lw %[b21], 4(%[addr_c]!) \n\t"
            "p.lw %[b22], 4(%[addr_c]!) \n\t"
            "p.lw %[b23], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b30], 4(%[addr_c]!) \n\t"
            "p.lw %[b31], 4(%[addr_c]!) \n\t"
            "p.lw %[b32], 4(%[addr_c]!) \n\t"
            "p.lw %[b33], %[c_incr](%[addr_c]!) \n\t"
            : [b00] "=&r"(bias00), [b01] "=&r"(bias01), [b02] "=&r"(bias02),
              [b03] "=&r"(bias03), [b10] "=&r"(bias10), [b11] "=&r"(bias11),
              [b12] "=&r"(bias12), [b13] "=&r"(bias13), [b20] "=&r"(bias20),
              [b21] "=&r"(bias21), [b22] "=&r"(bias22), [b23] "=&r"(bias23),
              [b30] "=&r"(bias30), [b31] "=&r"(bias31), [b32] "=&r"(bias32),
              [b33] "=&r"(bias33), [addr_c] "+&r"(idx_c)
            : [c_incr] "r"(P_incr)
            : "memory");

        sum00 = alpha * sum00 + beta * bias00 + bias;
        sum01 = alpha * sum01 + beta * bias01 + bias;
        sum02 = alpha * sum02 + beta * bias02 + bias;
        sum03 = alpha * sum03 + beta * bias03 + bias;
        sum10 = alpha * sum10 + beta * bias10 + bias;
        sum11 = alpha * sum11 + beta * bias11 + bias;
        sum12 = alpha * sum12 + beta * bias12 + bias;
        sum13 = alpha * sum13 + beta * bias13 + bias;
        sum20 = alpha * sum20 + beta * bias20 + bias;
        sum21 = alpha * sum21 + beta * bias21 + bias;
        sum22 = alpha * sum22 + beta * bias22 + bias;
        sum23 = alpha * sum23 + beta * bias23 + bias;
        sum30 = alpha * sum30 + beta * bias30 + bias;
        sum31 = alpha * sum31 + beta * bias31 + bias;
        sum32 = alpha * sum32 + beta * bias32 + bias;
        sum33 = alpha * sum33 + beta * bias33 + bias;

        int32_t _add0, _add1, _add2, _add3;
        int32_t _mul0, _mul1, _mul2, _mul3;
        if (per_row_quant) {
          __asm__ volatile(
              "p.lw %[add0], 4(%[addr_add]!) \n\t"
              "p.lw %[add1], 4(%[addr_add]!) \n\t"
              "p.lw %[add2], 4(%[addr_add]!) \n\t"
              "p.lw %[add3], 4(%[addr_add]!) \n\t"
              "p.lw %[mul0], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul1], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul2], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul3], 4(%[addr_mul]!) \n\t"
              : [add0] "=&r"(_add0), [mul0] "=&r"(_mul0), [add1] "=&r"(_add1),
                [mul1] "=&r"(_mul1), [add2] "=&r"(_add2), [mul2] "=&r"(_mul2),
                [add3] "=&r"(_add3), [mul3] "=&r"(_mul3),
                [addr_add] "+&r"(idx_add), [addr_mul] "+&r"(idx_mul)::"memory");
        } else {
          _add0 = add[0];
          _add1 = add[0];
          _add2 = add[0];
          _add3 = add[0];
          _mul0 = mul[0];
          _mul1 = mul[0];
          _mul2 = mul[0];
          _mul3 = mul[0];
        }

        sum00 = sum00 * _mul0 + rqs_bias + _add0;
        sum01 = sum01 * _mul0 + rqs_bias + _add0;
        sum02 = sum02 * _mul0 + rqs_bias + _add0;
        sum03 = sum03 * _mul0 + rqs_bias + _add0;
        sum10 = sum10 * _mul1 + rqs_bias + _add1;
        sum11 = sum11 * _mul1 + rqs_bias + _add1;
        sum12 = sum12 * _mul1 + rqs_bias + _add1;
        sum13 = sum13 * _mul1 + rqs_bias + _add1;
        sum20 = sum20 * _mul2 + rqs_bias + _add2;
        sum21 = sum21 * _mul2 + rqs_bias + _add2;
        sum22 = sum22 * _mul2 + rqs_bias + _add2;
        sum23 = sum23 * _mul2 + rqs_bias + _add2;
        sum30 = sum30 * _mul3 + rqs_bias + _add3;
        sum31 = sum31 * _mul3 + rqs_bias + _add3;
        sum32 = sum32 * _mul3 + rqs_bias + _add3;
        sum33 = sum33 * _mul3 + rqs_bias + _add3;

        sum00 = (sum00 >> log2D) + Y_offset;
        sum01 = (sum01 >> log2D) + Y_offset;
        sum02 = (sum02 >> log2D) + Y_offset;
        sum03 = (sum03 >> log2D) + Y_offset;
        sum10 = (sum10 >> log2D) + Y_offset;
        sum11 = (sum11 >> log2D) + Y_offset;
        sum12 = (sum12 >> log2D) + Y_offset;
        sum13 = (sum13 >> log2D) + Y_offset;
        sum20 = (sum20 >> log2D) + Y_offset;
        sum21 = (sum21 >> log2D) + Y_offset;
        sum22 = (sum22 >> log2D) + Y_offset;
        sum23 = (sum23 >> log2D) + Y_offset;
        sum30 = (sum30 >> log2D) + Y_offset;
        sum31 = (sum31 >> log2D) + Y_offset;
        sum32 = (sum32 >> log2D) + Y_offset;
        sum33 = (sum33 >> log2D) + Y_offset;

        sum0[0] = (int8_t)__CLIP(sum00, 7);
        sum0[1] = (int8_t)__CLIP(sum01, 7);
        sum0[2] = (int8_t)__CLIP(sum02, 7);
        sum0[3] = (int8_t)__CLIP(sum03, 7);
        sum1[0] = (int8_t)__CLIP(sum10, 7);
        sum1[1] = (int8_t)__CLIP(sum11, 7);
        sum1[2] = (int8_t)__CLIP(sum12, 7);
        sum1[3] = (int8_t)__CLIP(sum13, 7);
        sum2[0] = (int8_t)__CLIP(sum20, 7);
        sum2[1] = (int8_t)__CLIP(sum21, 7);
        sum2[2] = (int8_t)__CLIP(sum22, 7);
        sum2[3] = (int8_t)__CLIP(sum23, 7);
        sum3[0] = (int8_t)__CLIP(sum30, 7);
        sum3[1] = (int8_t)__CLIP(sum31, 7);
        sum3[2] = (int8_t)__CLIP(sum32, 7);
        sum3[3] = (int8_t)__CLIP(sum33, 7);

        __asm__ volatile("p.sw %[s0], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s1], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s2], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s3], %[y_incr](%[addr_y]!) \n\t"
                         : [addr_y] "+&r"(idx_y)
                         : [s0] "r"(sum0), [s1] "r"(sum1), [s2] "r"(sum2),
                           [s3] "r"(sum3), [y_incr] "r"(P)
                         : "memory");
        /* The asm code above implements the following commented C code */
        // *(idx_y) = sum0; idx_y += P;
        // *(idx_y) = sum1; idx_y += P;
        // *(idx_y) = sum2; idx_y += P;
        // *(idx_y) = sum3; idx_y += P;

        idx_a -= N * M - 4; // adjust A matrix pointer
      }
    }
  } else if (transA == 0 && transB == 1) {
    // Row decrement for A matrix
    int32_t const N_decr = -(int)N + 4;
    int32_t const B_decr = -(int)N * 3 + 4;
    // Row increment for C matrix
    uint32_t const P_incr = (P * 4) - 12;

    v4s aVecOffset = {(int8_t)A_offset, (int8_t)A_offset, (int8_t)A_offset,
                      (int8_t)A_offset};
    v4s bVecOffset = {(int8_t)B_offset, (int8_t)B_offset, (int8_t)B_offset,
                      (int8_t)B_offset};

    for (k = core_id; k < P / 4; k += numThreads) {
      const int8_t *idx_a = &pSrcA[0];      // start_a
      const int32_t *idx_c = &pSrcC[k * 4]; // start_c
      int8_t *idx_y = &pDstY[k * 4];        // start_y
      int8_t const *end_y = &pDstY[P * M];  // actually (P * M) + (k * 4)
      while (idx_y < end_y) {
        int32_t sum00 = 0;
        int32_t sum01 = 0;
        int32_t sum02 = 0;
        int32_t sum03 = 0;
        int32_t sum10 = 0;
        int32_t sum11 = 0;
        int32_t sum12 = 0;
        int32_t sum13 = 0;

        v4s sum0, sum1;

        int8_t const *end_a = idx_a + N;
        const int8_t *idx_b = &pSrcB[k * 4 * N]; // start_b
        while (idx_a < end_a) {
          v4s aVec0, aVec1;

          v4s bVec0, bVec1, bVec2, bVec3;

          __asm__ volatile(
              "p.lw %[a0], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[a1], %[a_decr](%[addr_a]!) \n\t"
              "p.lw %[b0], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[b1], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[b2], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[b3], %[b_decr](%[addr_b]!) \n\t"
              : [a0] "=&r"(aVec0), [a1] "=&r"(aVec1), [b0] "=&r"(bVec0),
                [b1] "=&r"(bVec1), [b2] "=&r"(bVec2), [b3] "=&r"(bVec3),
                [addr_a] "+&r"(idx_a), [addr_b] "+&r"(idx_b)
              : [a_incr] "r"(N), [a_decr] "r"(N_decr), [b_incr] "r"(N),
                [b_decr] "r"(B_decr)
              : "memory");
          /* The asm code above implements the following commented C code */
          // go to next row, same column
          // v4s aVec0 = *((v4s *)idx_a); idx_a += N;
          // go to previous row, one column forward
          // v4s aVec1 = *((v4s *)idx_a); idx_a -= N - 4;
          // v4s bVec0 = *((v4s *)idx_b); idx_b += N;
          // v4s bVec1 = *((v4s *)idx_b); idx_b += N;
          // v4s bVec2 = *((v4s *)idx_b); idx_b += N;
          // v4s bVec3 = *((v4s *)idx_b); idx_b -= 3*N - 4;
          aVec0 = __ADD4(aVec0, aVecOffset);
          aVec1 = __ADD4(aVec1, aVecOffset);

          bVec0 = __ADD4(bVec0, bVecOffset);
          bVec1 = __ADD4(bVec1, bVecOffset);
          bVec2 = __ADD4(bVec2, bVecOffset);
          bVec3 = __ADD4(bVec3, bVecOffset);

          sum00 = __SUMDOTP4(aVec0, bVec0, sum00);
          sum01 = __SUMDOTP4(aVec0, bVec1, sum01);
          sum02 = __SUMDOTP4(aVec0, bVec2, sum02);
          sum03 = __SUMDOTP4(aVec0, bVec3, sum03);
          sum10 = __SUMDOTP4(aVec1, bVec0, sum10);
          sum11 = __SUMDOTP4(aVec1, bVec1, sum11);
          sum12 = __SUMDOTP4(aVec1, bVec2, sum12);
          sum13 = __SUMDOTP4(aVec1, bVec3, sum13);
        }
        int32_t bias00, bias01, bias02, bias03;
        int32_t bias10, bias11, bias12, bias13;

        __asm__ volatile(
            "p.lw %[b00], 4(%[addr_c]!) \n\t"
            "p.lw %[b01], 4(%[addr_c]!) \n\t"
            "p.lw %[b02], 4(%[addr_c]!) \n\t"
            "p.lw %[b03], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b10], 4(%[addr_c]!) \n\t"
            "p.lw %[b11], 4(%[addr_c]!) \n\t"
            "p.lw %[b12], 4(%[addr_c]!) \n\t"
            "p.lw %[b13], %[c_incr](%[addr_c]!) \n\t"
            : [b00] "=&r"(bias00), [b01] "=&r"(bias01), [b02] "=&r"(bias02),
              [b03] "=&r"(bias03), [b10] "=&r"(bias10), [b11] "=&r"(bias11),
              [b12] "=&r"(bias12), [b13] "=&r"(bias13), [addr_c] "+&r"(idx_c)
            : [c_incr] "r"(P_incr)
            : "memory");

        sum00 = alpha * sum00 + beta * bias00 + bias;
        sum01 = alpha * sum01 + beta * bias01 + bias;
        sum02 = alpha * sum02 + beta * bias02 + bias;
        sum03 = alpha * sum03 + beta * bias03 + bias;
        sum10 = alpha * sum10 + beta * bias10 + bias;
        sum11 = alpha * sum11 + beta * bias11 + bias;
        sum12 = alpha * sum12 + beta * bias12 + bias;
        sum13 = alpha * sum13 + beta * bias13 + bias;

        int32_t _add0, _add1;
        int32_t _mul0, _mul1;
        if (per_row_quant) {
          __asm__ volatile(
              "p.lw %[add0], 4(%[addr_add]!) \n\t"
              "p.lw %[add1], 4(%[addr_add]!) \n\t"
              "p.lw %[mul0], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul1], 4(%[addr_mul]!) \n\t"
              : [add0] "=&r"(_add0), [mul0] "=&r"(_mul0), [add1] "=&r"(_add1),
                [mul1] "=&r"(_mul1), [addr_add] "+&r"(idx_add),
                [addr_mul] "+&r"(idx_mul)::"memory");
        } else {
          _add0 = add[0];
          _add1 = add[0];
          _mul0 = mul[0];
          _mul1 = mul[0];
        }

        sum00 = sum00 * _mul0 + rqs_bias + _add0;
        sum01 = sum01 * _mul0 + rqs_bias + _add0;
        sum02 = sum02 * _mul0 + rqs_bias + _add0;
        sum03 = sum03 * _mul0 + rqs_bias + _add0;
        sum10 = sum10 * _mul1 + rqs_bias + _add1;
        sum11 = sum11 * _mul1 + rqs_bias + _add1;
        sum12 = sum12 * _mul1 + rqs_bias + _add1;
        sum13 = sum13 * _mul1 + rqs_bias + _add1;

        sum00 = (sum00 >> log2D) + Y_offset;
        sum01 = (sum01 >> log2D) + Y_offset;
        sum02 = (sum02 >> log2D) + Y_offset;
        sum03 = (sum03 >> log2D) + Y_offset;
        sum10 = (sum10 >> log2D) + Y_offset;
        sum11 = (sum11 >> log2D) + Y_offset;
        sum12 = (sum12 >> log2D) + Y_offset;
        sum13 = (sum13 >> log2D) + Y_offset;

        sum0[0] = (int8_t)__CLIP(sum00, 7);
        sum0[1] = (int8_t)__CLIP(sum01, 7);
        sum0[2] = (int8_t)__CLIP(sum02, 7);
        sum0[3] = (int8_t)__CLIP(sum03, 7);
        sum1[0] = (int8_t)__CLIP(sum10, 7);
        sum1[1] = (int8_t)__CLIP(sum11, 7);
        sum1[2] = (int8_t)__CLIP(sum12, 7);
        sum1[3] = (int8_t)__CLIP(sum13, 7);

        __asm__ volatile("p.sw %[s0], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s1], %[y_incr](%[addr_y]!) \n\t"
                         : [addr_y] "+&r"(idx_y)
                         : [s0] "r"(sum0), [s1] "r"(sum1), [y_incr] "r"(P)
                         : "memory");
        /* The asm code above implements the following commented C code */
        // *(idx_y) = sum0; idx_y += P;
        // *(idx_y) = sum1; idx_y += P;

        idx_a += N; // adjust A matrix pointer
      }
    }
  } else if (transA == 1 && transB == 1) {
    // Masks for shuffles
    static v4s mask0 = {0, 1, 4, 5};
    static v4s mask1 = {2, 3, 6, 7};
    static v4s mask2 = {0, 2, 4, 6};
    static v4s mask3 = {1, 3, 5, 7};

    // Row decrement for A matrix
    int32_t const B_decr = -(int)N * 3 + 4;
    // Row increment for C matrix
    uint32_t const P_incr = (P * 4) - 12;

    v4s aVecOffset = {(int8_t)A_offset, (int8_t)A_offset, (int8_t)A_offset,
                      (int8_t)A_offset};
    v4s bVecOffset = {(int8_t)B_offset, (int8_t)B_offset, (int8_t)B_offset,
                      (int8_t)B_offset};

    for (k = core_id; k < P / 4; k += numThreads) {
      const int8_t *idx_a = &pSrcA[0];      // start_a
      const int32_t *idx_c = &pSrcC[k * 4]; // start_c
      int8_t *idx_y = &pDstY[k * 4];        // start_y
      int8_t const *end_y = &pDstY[P * M];  // actually (P * M) + (k * 4)
      while (idx_y < end_y) {
        int32_t sum00 = 0;
        int32_t sum01 = 0;
        int32_t sum02 = 0;
        int32_t sum03 = 0;
        int32_t sum10 = 0;
        int32_t sum11 = 0;
        int32_t sum12 = 0;
        int32_t sum13 = 0;
        int32_t sum20 = 0;
        int32_t sum21 = 0;
        int32_t sum22 = 0;
        int32_t sum23 = 0;
        int32_t sum30 = 0;
        int32_t sum31 = 0;
        int32_t sum32 = 0;
        int32_t sum33 = 0;

        v4s sum0, sum1, sum2, sum3;

        int8_t const *end_a = idx_a + N * M;
        const int8_t *idx_b = &pSrcB[k * 4 * N]; // start_b
        while (idx_a < end_a) {

          v4s bVec0, bVec1, bVec2, bVec3;
          v4s temp0, temp1, temp2, temp3;

          __asm__ volatile(
              "p.lw %[at0], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[at1], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[at2], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[at3], %[a_incr](%[addr_a]!) \n\t"
              "p.lw %[bt0], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[bt1], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[bt2], %[b_incr](%[addr_b]!) \n\t"
              "p.lw %[bt3], %[b_decr](%[addr_b]!) \n\t"
              : [at0] "=&r"(temp0), [at1] "=&r"(temp1), [at2] "=&r"(temp2),
                [at3] "=&r"(temp3), [bt0] "=&r"(bVec0), [bt1] "=&r"(bVec1),
                [bt2] "=&r"(bVec2), [bt3] "=&r"(bVec3), [addr_a] "+&r"(idx_a),
                [addr_b] "+&r"(idx_b)
              : [a_incr] "r"(M), [b_incr] "r"(N), [b_decr] "r"(B_decr)
              : "memory");
          /* The asm code above implements the following commented C code */
          // go to next row, same column
          // v4s aVec0 = *((v4s *)idx_a); idx_a += M;
          // v4s aVec1 = *((v4s *)idx_a); idx_a += M;
          // v4s aVec2 = *((v4s *)idx_a); idx_a += M;
          // v4s aVec3 = *((v4s *)idx_a); idx_a += M;
          // v4s bVec0 = *((v4s *)idx_b); idx_b += P;
          // v4s bVec1 = *((v4s *)idx_b); idx_b += P;
          // v4s bVec2 = *((v4s *)idx_b); idx_b += P;
          // v4s bVec3 = *((v4s *)idx_b); idx_b += P;

          bVec0 = __ADD4(bVec0, bVecOffset);
          bVec1 = __ADD4(bVec1, bVecOffset);
          bVec2 = __ADD4(bVec2, bVecOffset);
          bVec3 = __ADD4(bVec3, bVecOffset);

          // Shuffles to transpose at runtime the chunk extracted from A before
          // multiplying with B chunk temp0-3 variables needed because shuffles
          // use rD as source, but also modify it, thus we need a copy of their
          // content to use it twice in their original form
          v4s temp4 = __builtin_shuffle(temp0, temp1, mask0); // 0,1,4,5
          v4s temp5 = __builtin_shuffle(temp2, temp3, mask0); // 8,9,12,13
          v4s temp6 = __builtin_shuffle(temp0, temp1, mask1); // 2,3,6,7
          v4s temp7 = __builtin_shuffle(temp2, temp3, mask1); // 3,7,11,15

          v4s aVec0 = __builtin_shuffle(temp4, temp5, mask2); // 0,4,8,12
          v4s aVec1 = __builtin_shuffle(temp4, temp5, mask3); // 1,5,9,13
          v4s aVec2 = __builtin_shuffle(temp6, temp7, mask2); // 2,6,10,14
          v4s aVec3 = __builtin_shuffle(temp6, temp7, mask3); // 3,7,11,15

          aVec0 = __ADD4(aVec0, aVecOffset);
          aVec1 = __ADD4(aVec1, aVecOffset);
          aVec2 = __ADD4(aVec2, aVecOffset);
          aVec3 = __ADD4(aVec3, aVecOffset);

          sum00 = __SUMDOTP4(aVec0, bVec0, sum00);
          sum01 = __SUMDOTP4(aVec0, bVec1, sum01);
          sum02 = __SUMDOTP4(aVec0, bVec2, sum02);
          sum03 = __SUMDOTP4(aVec0, bVec3, sum03);
          sum10 = __SUMDOTP4(aVec1, bVec0, sum10);
          sum11 = __SUMDOTP4(aVec1, bVec1, sum11);
          sum12 = __SUMDOTP4(aVec1, bVec2, sum12);
          sum13 = __SUMDOTP4(aVec1, bVec3, sum13);
          sum20 = __SUMDOTP4(aVec2, bVec0, sum20);
          sum21 = __SUMDOTP4(aVec2, bVec1, sum21);
          sum22 = __SUMDOTP4(aVec2, bVec2, sum22);
          sum23 = __SUMDOTP4(aVec2, bVec3, sum23);
          sum30 = __SUMDOTP4(aVec3, bVec0, sum30);
          sum31 = __SUMDOTP4(aVec3, bVec1, sum31);
          sum32 = __SUMDOTP4(aVec3, bVec2, sum32);
          sum33 = __SUMDOTP4(aVec3, bVec3, sum33);
        }
        int32_t bias00, bias01, bias02, bias03;
        int32_t bias10, bias11, bias12, bias13;
        int32_t bias20, bias21, bias22, bias23;
        int32_t bias30, bias31, bias32, bias33;

        __asm__ volatile(
            "p.lw %[b00], 4(%[addr_c]!) \n\t"
            "p.lw %[b01], 4(%[addr_c]!) \n\t"
            "p.lw %[b02], 4(%[addr_c]!) \n\t"
            "p.lw %[b03], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b10], 4(%[addr_c]!) \n\t"
            "p.lw %[b11], 4(%[addr_c]!) \n\t"
            "p.lw %[b12], 4(%[addr_c]!) \n\t"
            "p.lw %[b13], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b20], 4(%[addr_c]!) \n\t"
            "p.lw %[b21], 4(%[addr_c]!) \n\t"
            "p.lw %[b22], 4(%[addr_c]!) \n\t"
            "p.lw %[b23], %[c_incr](%[addr_c]!) \n\t"
            "p.lw %[b30], 4(%[addr_c]!) \n\t"
            "p.lw %[b31], 4(%[addr_c]!) \n\t"
            "p.lw %[b32], 4(%[addr_c]!) \n\t"
            "p.lw %[b33], %[c_incr](%[addr_c]!) \n\t"
            : [b00] "=&r"(bias00), [b01] "=&r"(bias01), [b02] "=&r"(bias02),
              [b03] "=&r"(bias03), [b10] "=&r"(bias10), [b11] "=&r"(bias11),
              [b12] "=&r"(bias12), [b13] "=&r"(bias13), [b20] "=&r"(bias20),
              [b21] "=&r"(bias21), [b22] "=&r"(bias22), [b23] "=&r"(bias23),
              [b30] "=&r"(bias30), [b31] "=&r"(bias31), [b32] "=&r"(bias32),
              [b33] "=&r"(bias33), [addr_c] "+&r"(idx_c)
            : [c_incr] "r"(P_incr)
            : "memory");

        sum00 = alpha * sum00 + beta * bias00 + bias;
        sum01 = alpha * sum01 + beta * bias01 + bias;
        sum02 = alpha * sum02 + beta * bias02 + bias;
        sum03 = alpha * sum03 + beta * bias03 + bias;
        sum10 = alpha * sum10 + beta * bias10 + bias;
        sum11 = alpha * sum11 + beta * bias11 + bias;
        sum12 = alpha * sum12 + beta * bias12 + bias;
        sum13 = alpha * sum13 + beta * bias13 + bias;
        sum20 = alpha * sum20 + beta * bias20 + bias;
        sum21 = alpha * sum21 + beta * bias21 + bias;
        sum22 = alpha * sum22 + beta * bias22 + bias;
        sum23 = alpha * sum23 + beta * bias23 + bias;
        sum30 = alpha * sum30 + beta * bias30 + bias;
        sum31 = alpha * sum31 + beta * bias31 + bias;
        sum32 = alpha * sum32 + beta * bias32 + bias;
        sum33 = alpha * sum33 + beta * bias33 + bias;

        int32_t _add0, _add1, _add2, _add3;
        int32_t _mul0, _mul1, _mul2, _mul3;
        if (per_row_quant) {
          __asm__ volatile(
              "p.lw %[add0], 4(%[addr_add]!) \n\t"
              "p.lw %[add1], 4(%[addr_add]!) \n\t"
              "p.lw %[add2], 4(%[addr_add]!) \n\t"
              "p.lw %[add3], 4(%[addr_add]!) \n\t"
              "p.lw %[mul0], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul1], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul2], 4(%[addr_mul]!) \n\t"
              "p.lw %[mul3], 4(%[addr_mul]!) \n\t"
              : [add0] "=&r"(_add0), [mul0] "=&r"(_mul0), [add1] "=&r"(_add1),
                [mul1] "=&r"(_mul1), [add2] "=&r"(_add2), [mul2] "=&r"(_mul2),
                [add3] "=&r"(_add3), [mul3] "=&r"(_mul3),
                [addr_add] "+&r"(idx_add), [addr_mul] "+&r"(idx_mul)::"memory");
        } else {
          _add0 = add[0];
          _add1 = add[0];
          _add2 = add[0];
          _add3 = add[0];
          _mul0 = mul[0];
          _mul1 = mul[0];
          _mul2 = mul[0];
          _mul3 = mul[0];
        }

        sum00 = sum00 * _mul0 + rqs_bias + _add0;
        sum01 = sum01 * _mul0 + rqs_bias + _add0;
        sum02 = sum02 * _mul0 + rqs_bias + _add0;
        sum03 = sum03 * _mul0 + rqs_bias + _add0;
        sum10 = sum10 * _mul1 + rqs_bias + _add1;
        sum11 = sum11 * _mul1 + rqs_bias + _add1;
        sum12 = sum12 * _mul1 + rqs_bias + _add1;
        sum13 = sum13 * _mul1 + rqs_bias + _add1;
        sum20 = sum20 * _mul2 + rqs_bias + _add2;
        sum21 = sum21 * _mul2 + rqs_bias + _add2;
        sum22 = sum22 * _mul2 + rqs_bias + _add2;
        sum23 = sum23 * _mul2 + rqs_bias + _add2;
        sum30 = sum30 * _mul3 + rqs_bias + _add3;
        sum31 = sum31 * _mul3 + rqs_bias + _add3;
        sum32 = sum32 * _mul3 + rqs_bias + _add3;
        sum33 = sum33 * _mul3 + rqs_bias + _add3;

        sum00 = (sum00 >> log2D) + Y_offset;
        sum01 = (sum01 >> log2D) + Y_offset;
        sum02 = (sum02 >> log2D) + Y_offset;
        sum03 = (sum03 >> log2D) + Y_offset;
        sum10 = (sum10 >> log2D) + Y_offset;
        sum11 = (sum11 >> log2D) + Y_offset;
        sum12 = (sum12 >> log2D) + Y_offset;
        sum13 = (sum13 >> log2D) + Y_offset;
        sum20 = (sum20 >> log2D) + Y_offset;
        sum21 = (sum21 >> log2D) + Y_offset;
        sum22 = (sum22 >> log2D) + Y_offset;
        sum23 = (sum23 >> log2D) + Y_offset;
        sum30 = (sum30 >> log2D) + Y_offset;
        sum31 = (sum31 >> log2D) + Y_offset;
        sum32 = (sum32 >> log2D) + Y_offset;
        sum33 = (sum33 >> log2D) + Y_offset;

        sum0[0] = (int8_t)__CLIP(sum00, 7);
        sum0[1] = (int8_t)__CLIP(sum01, 7);
        sum0[2] = (int8_t)__CLIP(sum02, 7);
        sum0[3] = (int8_t)__CLIP(sum03, 7);
        sum1[0] = (int8_t)__CLIP(sum10, 7);
        sum1[1] = (int8_t)__CLIP(sum11, 7);
        sum1[2] = (int8_t)__CLIP(sum12, 7);
        sum1[3] = (int8_t)__CLIP(sum13, 7);
        sum2[0] = (int8_t)__CLIP(sum20, 7);
        sum2[1] = (int8_t)__CLIP(sum21, 7);
        sum2[2] = (int8_t)__CLIP(sum22, 7);
        sum2[3] = (int8_t)__CLIP(sum23, 7);
        sum3[0] = (int8_t)__CLIP(sum30, 7);
        sum3[1] = (int8_t)__CLIP(sum31, 7);
        sum3[2] = (int8_t)__CLIP(sum32, 7);
        sum3[3] = (int8_t)__CLIP(sum33, 7);

        __asm__ volatile("p.sw %[s0], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s1], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s2], %[y_incr](%[addr_y]!) \n\t"
                         "p.sw %[s3], %[y_incr](%[addr_y]!) \n\t"
                         : [addr_y] "+&r"(idx_y)
                         : [s0] "r"(sum0), [s1] "r"(sum1), [s2] "r"(sum2),
                           [s3] "r"(sum3), [y_incr] "r"(P)
                         : "memory");
        /* The asm code above implements the following commented C code */
        // *(idx_y) = sum0; idx_y += P;
        // *(idx_y) = sum1; idx_y += P;
        // *(idx_y) = sum2; idx_y += P;
        // *(idx_y) = sum3; idx_y += P;

        idx_a -= N * M - 4; // adjust A matrix pointer
      }
    }
  }
}

#endif //__XPULPIMG
