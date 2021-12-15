/* =====================================================================
 * Title:        DWConvolution.h
 * Description:
 *
 * Date:         05.01.2023
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

#ifndef __DEEPLOY_BASIC_MATH_DWCONVOLUTION_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_DWCONVOLUTION_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* This file implements depth-wise convolution.
 *
 * A is an M x N input matrix, B is a P x Q kernel matrix and C is and MxN
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

#endif //__DEEPLOY_BASIC_MATH_DWCONVOLUTION_KERNEL_HEADER_
