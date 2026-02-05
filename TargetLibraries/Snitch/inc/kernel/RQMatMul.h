/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_RQMATMUL_KERNEL_HEADER_
#define __DEEPLOY_MATH_RQMATMUL_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

/* This library implements the requantiyed matrix multiplication for several
 * data widths in multiple different ways. The functions all follow the
 * following format:
 *
 * A is an M x N matrix, B is a N x P matrix, and C is a M x P matrix
 * C = AB
 *
 * Note that all the matrices dimensions must be multiples of 4; these
 * kernels do not have clean-up code and remaining elements would not be
 * considered, leading to wrong results
 */

/******************************************************************************/
/*               Requantized Matrix Multiplication (8bit)                     */
/******************************************************************************/

/*
 * Matrix multiplication ----------------------------------
 * kernel     = RQMatMul_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = no
 * simd       = no
 * cleanup    = yes
 */
void RQMatMul_parallel_s8_rv32im(int8_t const *__restrict__ pSrcA,
                                 int8_t const *__restrict__ pSrcB,
                                 int8_t *__restrict__ pDstC, uint32_t M,
                                 uint32_t N, uint32_t P, int32_t *mul,
                                 int32_t *add, int32_t log2D, bool rounding,
                                 bool per_row_quant, int32_t A_offset,
                                 int32_t B_offset, int32_t output_offset,
                                 int8_t output_min, int8_t output_max);

/*
 * Matrix multiplication ----------------------------------
 * kernel     = RQMatMul_unrolled_2x2_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * simd       = no
 * cleanup    = no
 */
void RQMatMul_unrolled_2x2_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int8_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant);

/*
 * Matrix multiplication ----------------------------------
 * kernel     = RQMatMul_unrolled_2x2_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * simd       = no
 * cleanup    = no
 */
void RQMatMul_offset_unrolled_2x2_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int8_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset,
    int32_t output_offset);

// Mapper Functions
static inline void __attribute__((always_inline))
RQMatMul_parallel_s8(int8_t const *__restrict__ pSrcA,
                     int8_t const *__restrict__ pSrcB,
                     int8_t *__restrict__ pDstC, uint32_t M, uint32_t N,
                     uint32_t P, int32_t *mul, int32_t *add, int32_t log2D,
                     bool rounding, bool per_row_quant, int32_t A_offset,
                     int32_t B_offset, int32_t output_offset, int8_t output_min,
                     int8_t output_max) {
  RQMatMul_parallel_s8_rv32im(pSrcA, pSrcB, pDstC, M, N, P, mul, add, log2D,
                              rounding, per_row_quant, A_offset, B_offset,
                              output_offset, output_min, output_max);
}

static inline void __attribute__((always_inline))
RQMatMul_unrolled_2x2_parallel_s8(int8_t const *__restrict__ pSrcA,
                                  int8_t const *__restrict__ pSrcB,
                                  int8_t *__restrict__ pDstC, uint32_t M,
                                  uint32_t N, uint32_t P, int32_t *mul,
                                  int32_t *add, int32_t log2D, bool rounding,
                                  bool per_row_quant) {
  RQMatMul_unrolled_2x2_parallel_s8_rv32im(pSrcA, pSrcB, pDstC, M, N, P, mul,
                                           add, log2D, rounding, per_row_quant);
}

static inline void __attribute__((always_inline))
RQMatMul_offset_unrolled_2x2_parallel_s8(int8_t const *__restrict__ pSrcA,
                                         int8_t const *__restrict__ pSrcB,
                                         int8_t *__restrict__ pDstC, uint32_t M,
                                         uint32_t N, uint32_t P, int32_t *mul,
                                         int32_t *add, int32_t log2D,
                                         bool rounding, bool per_row_quant,
                                         int32_t A_offset, int32_t B_offset,
                                         int32_t output_offset) {
  RQMatMul_offset_unrolled_2x2_parallel_s8_rv32im(
      pSrcA, pSrcB, pDstC, M, N, P, mul, add, log2D, rounding, per_row_quant,
      A_offset, B_offset, output_offset);
}

#endif //__DEEPLOY_MATH_RQMATMUL_KERNEL_HEADER_
