/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_UNIFORMREQUANTSHIFT_KERNEL_HEADER_
#define __DEEPLOY_MATH_UNIFORMREQUANTSHIFT_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void UniformRequantShift_s8_s8(int8_t *data_in, int32_t size, int32_t mul,
                               int32_t add, int8_t *data_out, int32_t log2D,
                               int32_t HW, int32_t input_offset,
                               int32_t output_offset, int8_t output_min,
                               int8_t output_max, bool rounding);

void UniformRequantShift_u8_s8(uint8_t *data_in, int32_t size, int32_t mul,
                               int32_t add, int8_t *data_out, int32_t log2D,
                               int32_t HW, int32_t input_offset,
                               int32_t output_offset, int8_t output_min,
                               int8_t output_max, bool rounding);

void UniformRequantShift_s16_s8(int16_t *data_in, int32_t size, int32_t mul,
                                int32_t add, int8_t *data_out, int32_t log2D,
                                int32_t HW, int32_t input_offset,
                                int32_t output_offset, int8_t output_min,
                                int8_t output_max, bool rounding);

void UniformRequantShift_s32_s8(int32_t *data_in, int32_t size, int32_t mul,
                                int32_t add, int8_t *data_out, int32_t log2D,
                                int32_t HW, int32_t input_offset,
                                int32_t output_offset, int8_t output_min,
                                int8_t output_max, bool rounding);

#endif // __DEEPLOY_MATH_UNIFORMREQUANTSHIFT_KERNEL_HEADER_