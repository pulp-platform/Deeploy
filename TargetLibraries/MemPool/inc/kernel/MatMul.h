/* =====================================================================
 * Title:        MatMul.h
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

#ifndef __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MATMUL_KERNEL_HEADER_

#include "DeeployMath.h"

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
                               int32_t B_offset, int32_t output_offset,
                               uint32_t core_id, uint32_t numThreads);

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
                                            uint32_t M, uint32_t N, uint32_t P,
                                            uint32_t core_id,
                                            uint32_t numThreads);

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
    int32_t A_offset, int32_t B_offset, int32_t output_offset, uint32_t core_id,
    uint32_t numThreads);

#ifdef __XPULPIMG
/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_2x4_parallel_s8_xpulpv2
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 8 elements of C per iteration (2x4 chunks)
 * simd       = yes, Xpulpv2 intrinsics
 * cleanup    = no
 *
 * Original plp_mat_mult_s8p_xpulpv2 from pulp-dsp
 */
void MatMul_unrolled_2x4_parallel_s8_xpulpv2(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    int32_t A_offset, int32_t B_offset, int32_t output_offset, uint32_t core_id,
    uint32_t numThreads);

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_2x4_pincr_asm_parallel_s8_xpulpv2
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 8 elements of C per iteration (2x4 chunks)
 * simd       = yes, Xpulpv2 intrinsics
 * cleanup    = no
 * other      = using pointer incrementing instead of array
 *              indexing and loads/stores explicitly written
 *              in asm, for optimal register utilization
 *
 * Inspired from plp_mat_mult_s8p_xpulpv2 from pulp-dsp
 */
void MatMul_unrolled_2x4_pincr_asm_parallel_s8_xpulpv2(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    uint32_t core_id, uint32_t numThreads);

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_2x4_pincr_asm_parallel_s8_xpulpv2
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 8 elements of C per iteration (2x4 chunks)
 * simd       = yes, Xpulpv2 intrinsics
 * cleanup    = no
 * other      = using pointer incrementing instead of array
 *              indexing and loads/stores explicitly written
 *              in asm, for optimal register utilization
 *
 * Inspired from plp_mat_mult_s8p_xpulpv2 from pulp-dsp
 */
void MatMul_offset_unrolled_2x4_pincr_asm_parallel_s8_xpulpv2(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    int32_t A_offset, int32_t B_offset, int32_t output_offset, uint32_t core_id,
    uint32_t numThreads);

#endif

// Mapper Functions
static inline void __attribute__((always_inline))
MatMul_parallel_s8(int8_t const *__restrict__ pSrcA,
                   int8_t const *__restrict__ pSrcB,
                   int32_t *__restrict__ pDstC, uint32_t M, uint32_t N,
                   uint32_t P, int32_t A_offset, int32_t B_offset,
                   int32_t output_offset, uint32_t core_id,
                   uint32_t numThreads) {
  MatMul_parallel_s8_rv32im(pSrcA, pSrcB, pDstC, M, N, P, A_offset, B_offset,
                            output_offset, core_id, numThreads);
}

static inline void __attribute__((always_inline))
MatMul_unrolled_2x2_parallel_s8(int8_t const *__restrict__ pSrcA,
                                int8_t const *__restrict__ pSrcB,
                                int32_t *__restrict__ pDstC, uint32_t M,
                                uint32_t N, uint32_t P, uint32_t core_id,
                                uint32_t numThreads) {
#ifdef __XPULPIMG
  MatMul_unrolled_2x4_pincr_asm_parallel_s8_xpulpv2(pSrcA, pSrcB, pDstC, M, N,
                                                    P, core_id, numThreads);
#else
  MatMul_unrolled_2x2_parallel_s8_rv32im(pSrcA, pSrcB, pDstC, M, N, P, core_id,
                                         numThreads);
#endif
}

static inline void __attribute__((always_inline))
MatMul_offset_unrolled_2x2_parallel_s8(int8_t const *__restrict__ pSrcA,
                                       int8_t const *__restrict__ pSrcB,
                                       int32_t *__restrict__ pDstC, uint32_t M,
                                       uint32_t N, uint32_t P, int32_t A_offset,
                                       int32_t B_offset, int32_t output_offset,
                                       uint32_t core_id, uint32_t numThreads) {
#ifdef __XPULPIMG
  MatMul_offset_unrolled_2x4_pincr_asm_parallel_s8_xpulpv2(
      pSrcA, pSrcB, pDstC, M, N, P, A_offset, B_offset, output_offset, core_id,
      numThreads);
#else
  MatMul_offset_unrolled_2x2_parallel_s8_rv32im(
      pSrcA, pSrcB, pDstC, M, N, P, A_offset, B_offset, output_offset, core_id,
      numThreads);
#endif
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
                                             uint32_t M, uint32_t N, uint32_t P,
                                             uint32_t core_id,
                                             uint32_t numThreads);

#ifdef __XPULPIMG
/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_4x2_parallel_s16_xpulpv2
 * data type  = 16-bit integer
 * multi-core = yes
 * unrolling  = 8 elements of C per iteration (4x2 chunks)
 * simd       = yes, Xpulpv2 intrinsics
 * cleanup    = no
 *
 * Original plp_mat_mult_s16p_xpulpv2 from pulp-dsp
 */
void MatMul_unrolled_4x2_parallel_s16_xpulpv2(int16_t const *__restrict__ pSrcA,
                                              int16_t const *__restrict__ pSrcB,
                                              int32_t *__restrict__ pDstC,
                                              uint32_t M, uint32_t N,
                                              uint32_t P, uint32_t core_id,
                                              uint32_t numThreads);

/*
 * Matrix multiplication ----------------------------------
 * kernel     = MatMul_unrolled_4x2_pincr_asm_parallel_s16_xpulpv2
 * data type  = 16-bit integer
 * multi-core = yes
 * unrolling  = 8 elements of C per iteration (4x2 chunks)
 * simd       = yes, Xpulpv2 intrinsics
 * cleanup    = no
 * other      = using pointer incrementing instead of array
 *              indexing and loads/stores explicitly written
 *              in asm, for optimal register utilization
 *
 * Inspired from plp_mat_mult_s16p_xpulpv2 from pulp-dsp
 */
void MatMul_unrolled_4x2_pincr_asm_parallel_s16_xpulpv2(
    int16_t const *__restrict__ pSrcA, int16_t const *__restrict__ pSrcB,
    int32_t *__restrict__ pDstC, uint32_t M, uint32_t N, uint32_t P,
    uint32_t core_id, uint32_t numThreads);

#endif //__XPULPIMG

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
                                             uint32_t M, uint32_t N, uint32_t P,
                                             uint32_t core_id,
                                             uint32_t numThreads);

#ifdef __XPULPIMG

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
void MatMul_unrolled_2x2_parallel_s32_xpulpv2(int32_t const *__restrict__ pSrcA,
                                              int32_t const *__restrict__ pSrcB,
                                              int32_t *__restrict__ pDstC,
                                              uint32_t M, uint32_t N,
                                              uint32_t P, uint32_t core_id,
                                              uint32_t numThreads);
#endif //__XPULPIMG

#endif //__DEEPLOY_MATH_MATMUL_KERNEL_HEADER_
