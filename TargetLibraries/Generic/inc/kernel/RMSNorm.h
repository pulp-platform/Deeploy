/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_RMSNORM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RMSNORM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 *
 */

/******************************************************************************/
/*                             Layernorm (8bit)                               */
/******************************************************************************/

void iRMSnorm_s8_s8(int8_t *data_in, int8_t *data_out, int32_t *weight,
                    int32_t input_offset, int32_t size, int32_t lastDimLength,
                    int32_t log2D);

#endif //__DEEPLOY_BASIC_MATH_RMSNORM_KERNEL_HEADER_
