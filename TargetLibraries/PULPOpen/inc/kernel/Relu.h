/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_RELU_KERNEL_HEADER_
#define __DEEPLOY_MATH_RELU_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Relu_fp32_fp32(float32_t *input, float32_t *output, uint32_t size);

#endif // __DEEPLOY_MATH_RELU_KERNEL_HEADER_
