/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_SIGMOID_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SIGMOID_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * element wise sigmoid
 */

/******************************************************************************/
/*                             Sigmoid                                        */
/******************************************************************************/
void Sigmoid_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t size);

#endif //__DEEPLOY_BASIC_MATH_SIGMOID_KERNEL_HEADER_
