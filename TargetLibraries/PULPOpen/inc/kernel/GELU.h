/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_GELU_KERNEL_HEADER_
#define __DEEPLOY_MATH_GELU_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_GELU_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         int32_t dataSize);

void PULP_GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out,
                                 int32_t dataSize);

#endif // __DEEPLOY_MATH_GELU_KERNEL_HEADER_
