/* =====================================================================
 * Title:        RQGemm.h
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

#ifndef __DEEPLOY_MATH_RQGEMM_KERNEL_HEADER_
#define __DEEPLOY_MATH_RQGEMM_KERNEL_HEADER_

#include "DeeployMath.h"

/*
 * This library implements the matrix multiplication for several data widths
 * in multiple different ways. The functions all follow the following format:
 *
 * A is an M x N matrix, B is a N x P matrix, and C is a M x P matrix
 * A' = transpose(A) if transA else A
 * B' = transpose(B) if transB else B
 *
 * Y = alpha * A' * B' + beta * C
 *
 */

/******************************************************************************/
/*          General Requantized Matrix Multiplication (8bit)                  */
/******************************************************************************/

/*
 * General Requantized Matrix Multiplication ----------------------------------
 * kernel     = RQGemm_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = no
 * simd       = no
 * cleanup    = yes
 */
void RQGemm_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, int8_t output_min, int8_t output_max, uint32_t core_id,
    uint32_t numThreads);

/*
 * General Requantized Matrix multiplication ----------------------------------
 * kernel     = RQGemm_offset_unrolled_2x2_parallel_s8_rv32im
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * simd       = no
 * cleanup    = no
 */
void RQGemm_offset_unrolled_2x2_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, uint32_t core_id, uint32_t numThreads);

#ifdef __XPULPIMG
/*
 * General Requantized Matrix multiplication ----------------------------------
 * kernel     = RQGemm_offset_unrolled_4x4_pincr_asm_parallel_s8_xpulpv2
 * data type  = 8-bit integer
 * multi-core = yes
 * unrolling  = tranA=0, trabsB=0: 8 elements of C per iteration (2x4 chunks)
 *              tranA=1, trabsB=0: 16 elements of C per iteration (4x4 chunks)
 *              tranA=0, trabsB=1: 8 elements of C per iteration (2x4 chunks)
 *              tranA=1, trabsB=1: 16 elements of C per iteration (4x4 chunks)
 * simd       = yes, Xpulpv2 intrinsics
 * cleanup    = no
 * other      = using pointer incrementing instead of array
 *              indexing and loads/stores explicitly written
 *              in asm, for optimal register utilization
 */
void RQGemm_offset_unrolled_4x4_pincr_asm_parallel_s8_xpulpv2(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, uint32_t core_id, uint32_t numThreads);

#endif //__XPULPIMG

// Mapper Functions
static inline void __attribute__((always_inline)) RQGemm_parallel_s8(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, int8_t output_min, int8_t output_max, uint32_t core_id,
    uint32_t numThreads) {
  RQGemm_parallel_s8_rv32im(
      pSrcA, pSrcB, pSrcC, pDstY, M, N, P, alpha, beta, transA, transB, mul,
      add, log2D, rounding, per_row_quant, A_offset, B_offset, C_offset,
      Y_offset, output_min, output_max, core_id, numThreads);
}

// Mapper Functions
static inline void __attribute__((always_inline))
RQGemm_offset_unrolled_2x2_parallel_s8(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, uint32_t core_id, uint32_t numThreads) {
#ifdef __XPULPIMG
  RQGemm_offset_unrolled_4x4_pincr_asm_parallel_s8_xpulpv2(
      pSrcA, pSrcB, pSrcC, pDstY, M, N, P, alpha, beta, transA, transB, mul,
      add, log2D, rounding, per_row_quant, A_offset, B_offset, C_offset,
      Y_offset, core_id, numThreads);
#else
  RQGemm_offset_unrolled_2x2_parallel_s8_rv32im(
      pSrcA, pSrcB, pSrcC, pDstY, M, N, P, alpha, beta, transA, transB, mul,
      add, log2D, rounding, per_row_quant, A_offset, B_offset, C_offset,
      Y_offset, core_id, numThreads);
#endif
}

#endif //__DEEPLOY_MATH_RQGEMM_KERNEL_HEADER_
