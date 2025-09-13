/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
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
