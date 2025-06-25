/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_CONVOLUTION_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_CONVOLUTION_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* This file implements convolution.
 *
 * A is an M x N input matrix, B is a P x Q kernel matrix and C is and MxN
 * output matrix
 *
 */

/******************************************************************************/
/*                         General Convolution (8bit)                         */
/******************************************************************************/

/*
 * 2D Convolution  ----------------------------------
 * kernel      = Conv2d_s8_s8_s32_NCHW
 * layout      = NCHW
 * data type   = 8-bit integer
 * kernel size = generic
 * unrolling   = no
 * simd        = no
 */
void Conv2d_s8_s8_s32_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                           uint32_t H, uint32_t W,
                           int8_t const *__restrict__ pSrcB, uint32_t F,
                           uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
                           int32_t *__restrict__ pDstC, int32_t input_offset,
                           int32_t output_offset);

void Conv2d_fp32_fp32_fp32_NCHW(const float *__restrict__ pSrcA, uint32_t C,
                                uint32_t H_padded, uint32_t W_padded,
                                const float *__restrict__ pSrcB, uint32_t F,
                                uint32_t P, uint32_t Q, uint32_t SP,
                                uint32_t SQ, const float *__restrict__ pSrcBias,
                                const bool has_bias, float *__restrict__ pDstC);

void Conv1d_fp32_fp32_fp32(
    const float32_t *__restrict__ pSrcA, // Input: [C_in, W_in]
    uint32_t C_in, uint32_t W_in,
    const float32_t *__restrict__ pSrcB, // Weights: [C_out, C_in, K]
    uint32_t C_out, uint32_t K, uint32_t stride,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC, // Output: [C_out, W_out]
    uint32_t W_out);

#endif //__DEEPLOY_BASIC_MATH_CONVOLUTION_KERNEL_HEADER_
