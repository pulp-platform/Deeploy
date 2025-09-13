/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_RQIHARDSWISH_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RQIHARDSWISH_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                             Requantized Hardswish (8bit)                   */
/******************************************************************************/

void RQiHardswish_s8_s8(int8_t *input, int8_t *output, int32_t size,
                        int32_t one_over_six, int32_t three, int32_t six,
                        int32_t input_offset, int32_t output_offset,
                        int32_t mul, int32_t add, int32_t shift);

#endif // __DEEPLOY_BASIC_MATH_RQIHARDSWISH_KERNEL_HEADER_
