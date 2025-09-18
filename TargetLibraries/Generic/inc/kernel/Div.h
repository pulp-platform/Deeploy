/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_DIV_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_DIV_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This file implements the element-wise binary division.
 */

/******************************************************************************/
/*                              Division (32bit)                              */
/******************************************************************************/

void Div_s32_s32(int32_t *data_in_nom, int32_t *data_in_denom, int32_t size_nom,
                 int32_t size_denom, int32_t nomStep, int32_t denomStep,
                 int32_t *data_out, int32_t Delta, int32_t eps, int32_t eta);

void Div_fp32_fp32_fp32(float32_t *data_in_1, float32_t *data_in_2,
                        float32_t *data_out, int32_t size);

#endif //__DEEPLOY_BASIC_MATH_DIV_KERNEL_HEADER_
