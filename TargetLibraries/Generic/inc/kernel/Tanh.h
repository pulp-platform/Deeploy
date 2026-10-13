/*
 * SPDX-FileCopyrightText: 2026 EPFL, ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_TANH_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_TANH_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * element wise tanh
 */

/******************************************************************************/
/*                                Tanh                                        */
/******************************************************************************/
void tanh_fp32(float32_t const *__restrict__ pSrcA, uint32_t size,
               float32_t *__restrict__ pDstC);

#endif //__DEEPLOY_BASIC_MATH_TANH_KERNEL_HEADER_