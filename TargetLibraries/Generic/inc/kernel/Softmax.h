/* =====================================================================
 * Title:        Softmax.h
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

#ifndef __DEEPLOY_BASIC_MATH_SOFTMAX_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SOFTMAX_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This file implements various softmax kernels.
 */

/******************************************************************************/
/*                              Softmax (8bit)                                */
/******************************************************************************/

/**
 * @brief Approximate softmax implementation according to the I-BERT paper.
 * @see https://arxiv.org/abs/2101.01321
 *
 * @param data_in
 * @param data_out
 * @param size
 * @param lastDimLength
 * @param coeffA
 * @param coeffB
 * @param coeffC
 * @param log2
 * @param n_levels
 */
void Softmax_s8_s8(int8_t *data_in, int8_t *data_out, uint32_t size,
                   uint32_t lastDimLength, int32_t coeffA, int32_t coeffB,
                   int64_t coeffC, int32_t log2, uint32_t n_levels);

/**
 * @brief Approximate softmax implementation.
 *
 * @param pSrcA
 * @param pDstB
 * @param pBufN
 * @param size
 * @param lastDimLength
 * @param n_levels
 */
void ITAMax_s8(int8_t const *__restrict__ pSrcA, int8_t *__restrict__ pDstB,
               int8_t *__restrict__ pBufN, uint32_t size,
               uint32_t lastDimLength, uint32_t n_levels);

/**
 * @brief Approximate partial softmax implementation used in ITA.
 *
 * @param pSrcA
 * @param pDstB
 * @param size
 * @param lastDimLength
 * @param group_width
 * @param n_levels
 */
void ITAPartialMax_s8(int8_t const *__restrict__ pSrcA,
                      int8_t *__restrict__ pDstB, uint32_t size,
                      uint32_t lastDimLength, uint32_t group_width,
                      uint32_t n_levels);

#endif //__DEEPLOY_BASIC_MATH_SOFTMAX_KERNEL_HEADER_
