/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_LAYERNORM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_LAYERNORM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 *
 */

/******************************************************************************/
/*                             Layernorm (8bit)                               */
/******************************************************************************/

void Layernorm_s8_s8(int8_t *data_in, int8_t *data_out, int32_t *weight,
                     int32_t *bias, int32_t input_offset, int32_t size,
                     int32_t lastDimLength, int32_t log2D);

void Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         float32_t *scale, float32_t *bias, float32_t epsilon,
                         int32_t size, int32_t lastDimLength);

#endif //__DEEPLOY_BASIC_MATH_LAYERNORM_KERNEL_HEADER_
