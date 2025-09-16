/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_RQIHARDSWISH_KERNEL_HEADER_
#define __DEEPLOY_MATH_RQIHARDSWISH_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void RQiHardswish_s8_s8_plp(int8_t *input, int8_t *output, int32_t size,
                            int32_t one_over_six, int32_t three, int32_t six,
                            int32_t mul, int32_t add, int32_t shift);

#endif // __DEEPLOY_MATH_RQIHARDSWISH_KERNEL_HEADER_
