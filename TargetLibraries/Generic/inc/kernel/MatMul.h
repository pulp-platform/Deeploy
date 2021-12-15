/* =====================================================================
 * Title:        MatMul.h
 * Description:
 *
 * Date:         19.12.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Moritz Scherer, ETH Zurich
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

#ifndef __DEEPLOY_BASIC_MATH_MATMUL_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_MATMUL_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This library implements the matrix multiplication for several data widths
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
 * kernel     = MatMul_unrolled_2x2_s8_rv32im
 * data type  = 8-bit integer
 * unrolling  = 4 elements of C per iteration (2x2 chunks)
 * cleanup    = yes
 */
void MatMul_s8_s8_s32(int8_t const *__restrict__ pSrcA,
                      int8_t const *__restrict__ pSrcB,
                      int32_t *__restrict__ pDstC, uint32_t M, uint32_t N,
                      uint32_t P, int32_t A_offset, int32_t B_offset,
                      int32_t C_offset);

#endif //__DEEPLOY_BASIC_MATH_MATMUL_KERNEL_HEADER_
