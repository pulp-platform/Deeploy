/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
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
