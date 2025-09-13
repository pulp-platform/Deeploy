/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_ISOFTMAX_KERNEL_HEADER_
#define __DEEPLOY_MATH_ISOFTMAX_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

void SnitchSoftmax_u8_u8(uint8_t *data_in, uint8_t *data_out,
                         uint32_t *lastDimBuffer, uint32_t size,
                         uint32_t lastDimLength, int32_t coeffB, int32_t coeffC,
                         int32_t log2);
void StnichSoftmax_i8_u8(int8_t *data_in, uint8_t *data_out,
                         uint32_t *lastDimBuffer, uint32_t size,
                         uint32_t lastDimLength, int32_t coeffB, int32_t coeffC,
                         int32_t log2);

#endif // __DEEPLOY_MATH_ISOFTMAX_KERNEL_HEADER_
