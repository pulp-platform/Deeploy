/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_
#define __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

void softmax_fp32(float *input, float *output, int32_t ldI,
                  int32_t batch_offset, int32_t batch_size, int32_t seq_len,
                  int32_t input_samples);

#endif // #define __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_