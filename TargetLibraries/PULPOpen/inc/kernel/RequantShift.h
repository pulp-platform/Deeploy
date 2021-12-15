/* =====================================================================
 * Title:        RequantShift_s8.c
 * Description:
 *
 * Date:         19.12.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Moritz Scherer, ETH Zurich
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployPULPMath.h"

void RequantShift_u8_s8_NHWC(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding);

void RequantShift_u16_s8_NHWC(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

void RequantShift_u32_s8_NHWC(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

void RequantShift_u8_s8_NCHW(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding);

void RequantShift_u16_s8_NCHW(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

void RequantShift_u32_s8_NCHW(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

void RequantShift_u8_u8_NHWC(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding);

void RequantShift_u16_u8_NHWC(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);

void RequantShift_u32_u8_NHWC(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);

void RequantShift_u8_u8_NCHW(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding);

void RequantShift_u16_u8_NCHW(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);

void RequantShift_u32_u8_NCHW(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);

void RequantShift_s8_u8_NHWC(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding);

void RequantShift_s16_u8_NHWC(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);

void RequantShift_s32_u8_NHWC(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);

void RequantShift_s8_u8_NCHW(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding);

void RequantShift_s16_u8_NCHW(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);

void RequantShift_s32_u8_NCHW(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding);
