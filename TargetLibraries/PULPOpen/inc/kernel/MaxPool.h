/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_
#define __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_MaxPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                  uint32_t W, uint32_t H, uint32_t C,
                                  uint32_t Q, uint32_t P, uint32_t SQ,
                                  uint32_t SP, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right);

#endif // __DEEPLOY_MATH_MAXPOOL_KERNEL_HEADER_
