/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_
#define __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULPSoftmax_u8_u8(uint8_t *data_in, uint8_t *data_out,
                       uint32_t *lastDimBuffer, uint32_t size,
                       uint32_t lastDimLength, int32_t coeffB, int32_t coeffC,
                       int32_t log2);
void PULPSoftmax_i8_u8(int8_t *data_in, uint8_t *data_out,
                       uint32_t *lastDimBuffer, uint32_t size,
                       uint32_t lastDimLength, int32_t coeffB, int32_t coeffC,
                       int32_t log2);
void PULP_Softmax_fp32_fp32(float32_t *input, float32_t *output, uint32_t size,
                            uint32_t last_dim_length);

#endif // __DEEPLOY_MATH_SOFTMAX_KERNEL_HEADER_
