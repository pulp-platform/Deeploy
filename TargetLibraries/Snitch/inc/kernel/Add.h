/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_ADD_KERNEL_HEADER_
#define __DEEPLOY_MATH_ADD_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

void Add_fp32(float32_t *pIn1, float32_t *pIn2, float32_t *pOut, uint32_t size);

void Add_fp32_broadcast(float32_t *pIn1, float32_t *pIn2, float32_t *pOut,
                        uint32_t *out_shape, uint32_t *strides1,
                        uint32_t *strides2, uint32_t ndim, uint32_t size);

void Add_fp32_lastdim(float32_t *pIn1, float32_t *pIn2, float32_t *pOut,
                      uint32_t outer_size, uint32_t inner_size);

#endif // __DEEPLOY_MATH_ADD_KERNEL_HEADER_
