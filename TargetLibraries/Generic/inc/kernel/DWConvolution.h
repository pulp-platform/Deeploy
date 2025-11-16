/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_DWCONVOLUTION_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_DWCONVOLUTION_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* This file implements depth-wise convolution.
 *
 * A is an M x N input matrix, B is a P x Q kernel matrix and C is and M x N
 * output matrix
 *
 */

/******************************************************************************/
/*                   General Depth-Wise Convolution (8bit)                    */
/******************************************************************************/

/*
 * 2D Convolution  ----------------------------------
 * kernel      = DWConv2d_s8_s8_s32_NCHW
 * layout      = NCHW
 * data type   = 8-bit integer
 * kernel size = generic
 * unrolling   = no
 * simd        = no
 */
void DWConv2d_s8_s8_s32_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                             uint32_t H, uint32_t W,
                             int8_t const *__restrict__ pSrcB, uint32_t P,
                             uint32_t Q, uint32_t SP, uint32_t SQ,
                             int32_t *__restrict__ pDstC, int32_t input_offset,
                             int32_t output_offset);

/******************************************************************************/
/*                   General Depth-Wise Convolution (32bit)                   */
/******************************************************************************/
/*
 * 2D DW Convolution  --------------------------------
 * kernel      = DWConv2d_fp32_fp32_fp32_NCHW
 * layout      = NCHW
 * data type   = 32-bit float
 * kernel size = generic
 * unrolling   = no
 * simd        = no
 */
void DWConv2d_fp32_fp32_fp32_NCHW(
    const float32_t *__restrict__ pSrcA, uint32_t C, uint32_t H_padded,
    uint32_t W_padded, const float32_t *__restrict__ pSrcB, uint32_t F,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    const float32_t *__restrict__ pSrcBias, const bool has_bias,
    float32_t *__restrict__ pDstC);

#endif //__DEEPLOY_BASIC_MATH_DWCONVOLUTION_KERNEL_HEADER_
