/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

/* This library implements the matrix multiplication for several data widths
 * in multiple different ways. The functions all follow the following format:
 *
 * A is an M x N matrix, B is a N x P matrix, and C is a M x P matrix
 * C = AB
 *
 * Note that all the matrices dimensions must be multiples of 4; these
 * kernels do not have clean-up code and remaining elements would not be
 * considered, leading to wrong results
 */

/******************************************************************************/
/*                         Matrix Multiplication (8bit)                       */
/******************************************************************************/

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = no
 * simd       = no
 * cleanup    = yes
 */
void MatMul_parallel_s8_rv32im(int8_t const *__restrict__ pSrcA,
                               int8_t const *__restrict__ pSrcB,
                               int32_t *__restrict__ pDstC, uint32_t M,
                               uint32_t N, uint32_t P, int32_t A_offset,
                               int32_t B_offset, int32_t output_offset);

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_2x2_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * simd       = no
 * cleanup    = no
 */
void MatMul_unrolled_2x2_parallel_s8_rv32im(int8_t const *__restrict__ pSrcA,
                                            int8_t const *__restrict__ pSrcB,
                                            int32_t *__restrict__ pDstC,
                                            uint32_t M, uint32_t N, uint32_t P);

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_2x2_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * simd       = no
 * cleanup    = no
 */
void MatMul_offset_unrolled_2x2_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    int32_t A_offset, int32_t B_offset, int32_t output_offset);

// Mapper Functions
static inline void __attribute__((always_inline))
MatMul_parallel_s8(int8_t const *__restrict__ pSrcA,
                   int8_t const *__restrict__ pSrcB,
                   int32_t *__restrict__ pDstC, uint32_t M, uint32_t N,
                   uint32_t P, int32_t A_offset, int32_t B_offset,
                   int32_t output_offset) {
  MatMul_parallel_s8_rv32im(pSrcA, pSrcB, pDstC, M, N, P, A_offset, B_offset,
                            output_offset);
}

static inline void __attribute__((always_inline))
MatMul_unrolled_2x2_parallel_s8(int8_t const *__restrict__ pSrcA,
                                int8_t const *__restrict__ pSrcB,
                                int32_t *__restrict__ pDstC, uint32_t M,
                                uint32_t N, uint32_t P) {
  MatMul_unrolled_2x2_parallel_s8_rv32im(pSrcA, pSrcB, pDstC, M, N, P);
}

static inline void __attribute__((always_inline))
MatMul_offset_unrolled_2x2_parallel_s8(int8_t const *__restrict__ pSrcA,
                                       int8_t const *__restrict__ pSrcB,
                                       int32_t *__restrict__ pDstC, uint32_t M,
                                       uint32_t N, uint32_t P, int32_t A_offset,
                                       int32_t B_offset,
                                       int32_t output_offset) {
  MatMul_offset_unrolled_2x2_parallel_s8_rv32im(
      pSrcA, pSrcB, pDstC, M, N, P, A_offset, B_offset, output_offset);
}

/******************************************************************************/
/*                        Matrix Multiplication (16bit)                       */
/******************************************************************************/

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_2x2_parallel_s16_rv32im
 * data type  = 16-bit integer
 * multi-core = yes
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * simd       = no
 * cleanup    = no
 */
void MatMul_unrolled_2x2_parallel_s16_rv32im(int16_t const *__restrict__ pSrcA,
                                             int16_t const *__restrict__ pSrcB,
                                             int32_t *__restrict__ pDstC,
                                             uint32_t M, uint32_t N,
                                             uint32_t P);

/******************************************************************************/
/*                        Matrix Multiplication (32bit)                       */
/******************************************************************************/

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_2x2_parallel_s32_xpulpv2
 * data type  = 32-bit integer
 * multi-core = yes
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * simd       = no
 * cleanup    = no
 * other      = loads/stores explicitly written in asm
 *              for optimal register utilization
 */
void MatMul_unrolled_2x2_parallel_s32_rv32im(int32_t const *__restrict__ pSrcA,
                                             int32_t const *__restrict__ pSrcB,
                                             int32_t *__restrict__ pDstC,
                                             uint32_t M, uint32_t N,
                                             uint32_t P);

#endif //__DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
