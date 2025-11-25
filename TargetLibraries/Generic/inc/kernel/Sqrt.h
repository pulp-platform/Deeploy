/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_SQRT_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SQRT_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * Square root operation - computes sqrt for each element
 */

/******************************************************************************/
/*                              Sqrt                                          */
/******************************************************************************/

void Sqrt_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t size);

void Sqrt_fp16_fp16(float16_t *data_in, float16_t *data_out, int32_t size);

#endif //__DEEPLOY_BASIC_MATH_SQRT_KERNEL_HEADER_
