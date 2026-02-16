/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_MAXPOOL_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_MAXPOOL_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* This file implements the MaxPool operation.
 *
 * A is an M x N input matrix, P x Q the kernel size and SPxSQ the kernel
 * stride.
 *
 */

/******************************************************************************/
/*                         General MaxPool (8bit)                         */
/******************************************************************************/

/*
 * 2D Maxpool  ----------------------------------
 * kernel      = MaxPool2d_s8_s8_NCHW
 * layout      = NCHW
 * data type   = 8-bit integer
 * kernel size = generic
 * unrolling   = no
 * simd        = no
 */
void MaxPool2d_s8_s8_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                          uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                          uint32_t SP, uint32_t SQ, int8_t *__restrict__ pDstC,
                          int32_t input_offset, int32_t output_offset);

void MaxPool2d_fp32_fp32_NCHW(float32_t const *__restrict__ pSrcA, uint32_t C,
                              uint32_t H, uint32_t W, uint32_t P, uint32_t Q,
                              uint32_t SP, uint32_t SQ,
                              float32_t *__restrict__ pDstC);

void MaxPool1d_s8_s8(int8_t const *__restrict__ pSrcA, uint32_t C, uint32_t L,
                     uint32_t K, uint32_t S, int8_t *__restrict__ pDstC,
                     int32_t input_offset, int32_t output_offset);

void MaxPool1d_fp32_fp32(float32_t const *__restrict__ pSrcA, uint32_t C,
                         uint32_t W, uint32_t K, uint32_t S,
                         float32_t *__restrict__ pDstC);

#endif //__DEEPLOY_BASIC_MATH_MAXPOOL_KERNEL_HEADER_
