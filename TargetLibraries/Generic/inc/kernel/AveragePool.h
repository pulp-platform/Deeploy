/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_AVERAGEPOOL_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_AVERAGEPOOL_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                                Average Pool                                */
/******************************************************************************/
void AveragePool2d_fp32_fp32(float32_t const *__restrict__ src,
                             float32_t *__restrict__ dst, uint32_t N,
                             uint32_t C, uint32_t H, uint32_t W,
                             uint32_t kernel_h, uint32_t kernel_w,
                             uint32_t stride_h, uint32_t stride_w,
                             uint32_t pad_top, uint32_t pad_left,
                             uint32_t pad_bottom, uint32_t pad_right);

void AveragePool1d_fp32_fp32(float32_t const *__restrict__ src,
                             float32_t *__restrict__ dst, uint32_t N,
                             uint32_t C, uint32_t L, uint32_t kernel_len,
                             uint32_t stride, uint32_t pad_left,
                             uint32_t pad_right);

#endif //__DEEPLOY_BASIC_MATH_AVERAGEPOOL_KERNEL_HEADER_
