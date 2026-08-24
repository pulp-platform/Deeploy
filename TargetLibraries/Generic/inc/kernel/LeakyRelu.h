/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_LEAKYRELU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_LEAKYRELU_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * element wise LeakyRelu function
 */

/******************************************************************************/
/*                          LeakyRelu                                         */
/******************************************************************************/
void LeakyRelu_fp32_fp32(const float32_t *data_in, float32_t *data_out,
                         int32_t size, float32_t alpha);

#endif //__DEEPLOY_BASIC_MATH_LEAKYRELU_KERNEL_HEADER_
