/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_GLOBALAVERAGEPOOL_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_GLOBALAVERAGEPOOL_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                                Average Pool                                */
/******************************************************************************/
void GlobalAveragePool_fp32_fp32(float32_t const *__restrict__ src,
                                 float32_t *__restrict__ dst, uint32_t N,
                                 uint32_t C, uint32_t spatial_size);

#endif //__DEEPLOY_BASIC_MATH_GLOBALAVERAGEPOOL_KERNEL_HEADER_
