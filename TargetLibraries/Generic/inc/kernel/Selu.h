/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_SELU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SELU_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * element wise Scaled Exponential Linear Unit (SELU) function
 */

/******************************************************************************/
/*                             Selu                                          */
/******************************************************************************/
void Selu_fp32_fp32(const float32_t *input, float32_t *output, int32_t size,
                    float32_t alpha, float32_t gamma);

#endif //__DEEPLOY_BASIC_MATH_SELU_KERNEL_HEADER_
