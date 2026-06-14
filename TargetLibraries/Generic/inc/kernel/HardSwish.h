/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_HARDSWISH_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_HARDSWISH_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                             Hardswish (8bit)                               */
/******************************************************************************/

void iHardswish_s8_s32(int8_t *input, int32_t *output, int32_t size,
                       int32_t one_over_six, int32_t three, int32_t six,
                       int32_t input_offset);

/******************************************************************************/
/*                             Hardswish (fp32)                               */
/******************************************************************************/

void HardSwish_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t size);

#endif // __DEEPLOY_BASIC_MATH_HARDSWISH_KERNEL_HEADER_