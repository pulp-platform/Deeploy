/* =====================================================================
 * Title:        Softmax.h
 * Description:
 *
 * Date:         25.04.2023
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

#ifndef __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_
#define __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_

#include "DeeployMath.h"

/*
 * This file implements various softmax kernels.
 */

/******************************************************************************/
/*                              Softmax (8bit)                                */
/******************************************************************************/

/**
 * @brief Approximate softmax implementation used in ITA.
 *
 * @param pSrcA
 * @param pDstB
 * @param pBufN
 * @param size
 * @param lastDimLength
 */
void ITAMax_parallel_s8(int8_t const *__restrict__ pSrcA,
                        int8_t *__restrict__ pDstB, int8_t *__restrict__ pBufN,
                        uint32_t size, uint32_t lastDimLength,
                        uint32_t n_levels, uint32_t core_id,
                        uint32_t numThreads);

#endif //__DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_
