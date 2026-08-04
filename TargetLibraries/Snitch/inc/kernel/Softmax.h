/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_
#define __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

void Softmax_fp32(float32_t *input, float32_t *output, uint32_t size,
                  uint32_t lastDimLength);

#endif // #define __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_
