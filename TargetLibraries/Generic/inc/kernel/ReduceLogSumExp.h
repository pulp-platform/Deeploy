/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_REDUCELOGSUMEXP_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_REDUCELOGSUMEXP_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void ReduceLogSumExp_fp32_fp32(float32_t *input, float32_t *output,
                               uint32_t outer_size, uint32_t axis_length,
                               uint32_t inner_size);

#endif // __DEEPLOY_BASIC_MATH_REDUCELOGSUMEXP_KERNEL_HEADER_
