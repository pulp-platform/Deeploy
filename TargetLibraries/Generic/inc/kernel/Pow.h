/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * This file implements the element-wise binary power operation.
 */

#ifndef __DEEPLOY_MATH_POW_KERNEL_HEADER_
#define __DEEPLOY_MATH_POW_KERNEL_HEADER_

#include "DeeployBasicMath.h"

void Pow_fp32_fp32_fp32(const float32_t *__restrict__ data_in,
                        const float32_t *__restrict__ exponent,
                        float32_t *__restrict__ data_out, int32_t size);

void Pow_fp32_scalar_fp32(const float32_t *__restrict__ data_in,
                          float32_t exponent, float32_t *__restrict__ data_out,
                          int32_t size);

#endif
