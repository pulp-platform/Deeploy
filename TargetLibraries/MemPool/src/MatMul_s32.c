/* =====================================================================
 * Title:        MatMul_s32.c
 * Description:
 *
 * Date:         29.11.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Samuel Riedel, ETH Zurich
 * - Sergio Mazzola, ETH Zurich
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

void MatMul_unrolled_2x2_parallel_s32_rv32im(int32_t const *__restrict__ pSrcA,
                                             int32_t const *__restrict__ pSrcB,
                                             int32_t *__restrict__ pDstC,
                                             uint32_t M, uint32_t N, uint32_t P,
                                             uint32_t core_id,
                                             uint32_t numThreads) {
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
        int32_t val_a00 = pSrcA[(i + 0) * N + k + 0];
        int32_t val_a01 = pSrcA[(i + 0) * N + k + 1];
        int32_t val_a10 = pSrcA[(i + 1) * N + k + 0];
        int32_t val_a11 = pSrcA[(i + 1) * N + k + 1];
        int32_t val_b00 = pSrcB[(k + 0) * P + j + 0];
        int32_t val_b01 = pSrcB[(k + 0) * P + j + 1];
        int32_t val_b10 = pSrcB[(k + 1) * P + j + 0];
        int32_t val_b11 = pSrcB[(k + 1) * P + j + 1];
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

#ifdef __XPULPIMG

void MatMul_unrolled_2x2_parallel_s32_xpulpv2(int32_t const *__restrict__ pSrcA,
                                              int32_t const *__restrict__ pSrcB,
                                              int32_t *__restrict__ pDstC,
                                              uint32_t M, uint32_t N,
                                              uint32_t P, uint32_t core_id,
                                              uint32_t numThreads) {
  // Parallelize by assigning each core one row
  uint32_t const c = 8; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);

  uint32_t const A_incr = (N * sizeof(int32_t)) - sizeof(int32_t);
  uint32_t const B_incr = (P * sizeof(int32_t)) - sizeof(int32_t);

  for (uint32_t i = 2 * (core_id / c); i < M; i += 2 * (numThreads / c)) {
    for (uint32_t j = c_start; j < c_end; j += 2) {
      int32_t c00 = 0;
      int32_t c01 = 0;
      int32_t c10 = 0;
      int32_t c11 = 0;

      for (uint32_t k = 0; k < N; k += 2) {
        const int32_t *idx_a = &pSrcA[i * N + k];
        const int32_t *idx_b = &pSrcB[k * P + j];
        int32_t val_a00, val_a01, val_a10, val_a11, val_b00, val_b01, val_b10,
            val_b11;
        __asm__ volatile(
            "p.lw %[a00], 4(%[addr_a]!) \n\t"
            "p.lw %[a01], %[a_incr](%[addr_a]!) \n\t"
            "p.lw %[a10], 4(%[addr_a]!) \n\t"
            "p.lw %[a11], 0(%[addr_a]) \n\t"
            "p.lw %[b00], 4(%[addr_b]!) \n\t"
            "p.lw %[b01], %[b_incr](%[addr_b]!) \n\t"
            "p.lw %[b10], 4(%[addr_b]!) \n\t"
            "p.lw %[b11], 0(%[addr_b]) \n\t"
            : [a00] "=&r"(val_a00), [a01] "=&r"(val_a01), [a10] "=&r"(val_a10),
              [a11] "=&r"(val_a11), [b00] "=&r"(val_b00), [b01] "=&r"(val_b01),
              [b10] "=&r"(val_b10), [b11] "=&r"(val_b11), [addr_a] "+&r"(idx_a),
              [addr_b] "+&r"(idx_b)
            : [a_incr] "r"(A_incr), [b_incr] "r"(B_incr)
            : "memory");
        /* The asm code above implements the following commented C code */
        // int32_t val_a00 =pSrcA[(i + 0) * N + k + 0];
        // int32_t val_a01 =pSrcA[(i + 0) * N + k + 1];
        // int32_t val_a10 =pSrcA[(i + 1) * N + k + 0];
        // int32_t val_a11 =pSrcA[(i + 1) * N + k + 1];
        // int32_t val_b00 =pSrcB[(k + 0) * P + j + 0];
        // int32_t val_b01 =pSrcB[(k + 0) * P + j + 1];
        // int32_t val_b10 =pSrcB[(k + 1) * P + j + 0];
        // int32_t val_b11 =pSrcB[(k + 1) * P + j + 1];
        c00 += val_a00 * val_b00;
        c00 += val_a01 * val_b10;
        c01 += val_a00 * val_b01;
        c01 += val_a01 * val_b11;
        c10 += val_a10 * val_b00;
        c10 += val_a11 * val_b10;
        c11 += val_a10 * val_b01;
        c11 += val_a11 * val_b11;
      }
      int32_t *idx_c = &pDstC[i * P + j];
      __asm__ volatile("p.sw %[s00], 4(%[addr_c]!) \n\t"
                       "p.sw %[s01], %[c_incr](%[addr_c]!) \n\t"
                       "p.sw %[s10], 4(%[addr_c]!) \n\t"
                       "p.sw %[s11], 0(%[addr_c]) \n\t"
                       : [addr_c] "+&r"(idx_c)
                       : [s00] "r"(c00), [s01] "r"(c01), [s10] "r"(c10),
                         [s11] "r"(c11), [c_incr] "r"(B_incr)
                       : "memory");
      /* The asm code above implements the following commented C code */
      // pDstC[(i + 0) * P + j + 0] = c00;
      // pDstC[(i + 0) * P + j + 1] = c01;
      // pDstC[(i + 1) * P + j + 0] = c10;
      // pDstC[(i + 1) * P + j + 1] = c11;
    }
  }
}
#endif //__XPULPIMG
