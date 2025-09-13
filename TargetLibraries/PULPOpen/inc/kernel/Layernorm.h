/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER_
#define __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias,
                              float32_t epsilon, uint32_t size,
                              uint32_t lastDimLength);

#endif // __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER__
