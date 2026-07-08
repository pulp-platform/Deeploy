/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_ADD_KERNEL_HEADER_
#define __DEEPLOY_MATH_ADD_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

void Add_fp32(float32_t *input1, float32_t *input2, float32_t *output,
              uint32_t size, uint32_t is_scalar);

#endif // __DEEPLOY_MATH_ADD_KERNEL_HEADER_
