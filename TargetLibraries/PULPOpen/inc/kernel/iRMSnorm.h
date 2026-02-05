/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_IRMSNORM_KERNEL_HEADER_
#define __DEEPLOY_MATH_IRMSNORM_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void iRMSnorm_s8_s8_plp(int8_t *data_in, int8_t *data_out, int32_t *weight,
                        int32_t size, int32_t lastDimLength, int32_t log2D);

#endif // __DEEPLOY_MATH_IRMSNORM_KERNEL_HEADER__
