/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This file implements the element-wise binary power operation.
 */

/******************************************************************************/
/*                                Power (32bit)                               */
/******************************************************************************/

#ifndef __DEEPLOY_MATH_POW_KERNEL_HEADER_
#define __DEEPLOY_MATH_POW_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Pow_fp32_int32_fp32(float32_t *data_in, int32_t exponent,
                         float32_t *data_out, int32_t size);

void Pow_fp16_int32_fp16(float16_t *data_in, int32_t exponent,
                         float16_t *data_out, int32_t size);
#endif
