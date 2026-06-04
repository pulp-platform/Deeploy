/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MUL_FP32_KERNEL_HEADER_
#define __DEEPLOY_MATH_MUL_FP32_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

/*
 * Element-wise Multiplication (FP32) with optional scalar broadcasting.
 *
 * is_scalar == 0:  output[i] = input1[i] * input2[i]
 * is_scalar != 0:  output[i] = input1[i] * input2[0]   (input2 read as scalar)
 *
 * input1:    First input tensor (float32)
 * input2:    Second input tensor (float32). Only input2[0] is read when
 *            is_scalar != 0.
 * output:    Output tensor (same shape as input1)
 * size:      Total number of elements
 * is_scalar: Non-zero selects the scalar-broadcast branch.
 *
 * multi-core      = yes
 * parallelization = element-wise
 */
void Mul_fp32(float32_t *input1, float32_t *input2, float32_t *output,
              uint32_t size, uint32_t is_scalar);

#endif // __DEEPLOY_MATH_MUL_FP32_KERNEL_HEADER_
