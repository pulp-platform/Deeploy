/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MUL_FP32_KERNEL_HEADER_
#define __DEEPLOY_MATH_MUL_FP32_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

/*
 * Element-wise Multiplication (FP32)
 *
 * Computes: output[i] = input1[i] * input2[i]
 *
 * input1:         First input tensor (float32)
 * input2:         Second input tensor (float32)
 * output:         Output tensor (same shape as input1)
 * size:           Total number of elements
 *
 * multi-core      = yes
 * parallelization = element-wise
 */
void Mul_fp32(float32_t *input1, float32_t *input2, float32_t *output,
              uint32_t size);

/*
 * Element-wise Multiplication with scalar broadcasting (FP32)
 *
 * Computes: output[i] = input1[i] * scalar
 *
 * input1:         Input tensor (float32)
 * scalar:         Scalar multiplier (float32)
 * output:         Output tensor (same shape as input1)
 * size:           Total number of elements
 *
 * multi-core      = yes
 * parallelization = element-wise
 */
void Mul_fp32_scalar(float32_t *input1, float32_t scalar, float32_t *output,
                     uint32_t size);

#endif // __DEEPLOY_MATH_MUL_FP32_KERNEL_HEADER_
