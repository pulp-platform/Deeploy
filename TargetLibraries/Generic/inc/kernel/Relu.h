/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_RELU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RELU_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Relu_fp32_fp32(float32_t *input, float32_t *output, int32_t size);

#endif // __DEEPLOY_BASIC_MATH_RELU_KERNEL_HEADER_