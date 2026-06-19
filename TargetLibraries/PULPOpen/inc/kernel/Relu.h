/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_RELU_KERNEL_HEADER_
#define __DEEPLOY_MATH_RELU_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Relu_fp32_fp32(float32_t *input, float32_t *output, uint32_t size);
void PULP_Relu6_fp32_fp32(float32_t *input, float32_t *output, uint32_t size);

void PULP_ReluGrad_fp32_fp32(float32_t *grad_in, float32_t *data_in, float32_t *grad_out, uint32_t size);

#endif // __DEEPLOY_MATH_RELU_KERNEL_HEADER_