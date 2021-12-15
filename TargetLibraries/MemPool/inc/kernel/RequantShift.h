/* =====================================================================
 * Title:        RequantShift.h
 * Description:
 *
 * Date:         24.04.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
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

#ifndef __DEEPLOY_MATH_REQUANTSHIFT_KERNEL_HEADER_
#define __DEEPLOY_MATH_REQUANTSHIFT_KERNEL_HEADER_

#include "DeeployMath.h"

/*
 * This file implements the requantization kernel for several data widths
 * in multiple different ways.
 */

/******************************************************************************/
/*                         Requantization to 8bit                             */
/******************************************************************************/

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_parallel_s8_s8_NHWC
 * layout           = NHWC
 * input data type  = 8-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = no
 * simd             = no
 */
void RequantShift_parallel_s8_s8_NHWC(int8_t *data_in, uint32_t size,
                                      int32_t *mul, int32_t *add,
                                      int8_t *data_out, int32_t log2D,
                                      uint32_t channels, int32_t input_offset,
                                      int32_t output_offset, int8_t output_min,
                                      int8_t output_max, bool rounding,
                                      uint32_t core_id, uint32_t numThreads);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_parallel_s16_s8_NHWC
 * layout           = NHWC
 * input data type  = 16-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = no
 * simd             = no
 */
void RequantShift_parallel_s16_s8_NHWC(int16_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t channels, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_parallel_s32_s8_NHWC
 * layout           = NHWC
 * input data type  = 32-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = no
 * simd             = no
 */
void RequantShift_parallel_s32_s8_NHWC(int32_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t channels, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_parallel_s16_s8_NCHW
 * layout           = NCHW
 * input data type  = 16-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = no
 * simd             = no
 */
void RequantShift_parallel_s8_s8_NCHW(int8_t *data_in, uint32_t size,
                                      int32_t *mul, int32_t *add,
                                      int8_t *data_out, int32_t log2D,
                                      uint32_t HW, int32_t input_offset,
                                      int32_t output_offset, int8_t output_min,
                                      int8_t output_max, bool rounding,
                                      uint32_t core_id, uint32_t numThreads);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_parallel_s16_s8_NCHW
 * layout           = NCHW
 * input data type  = 16-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = no
 * simd             = no
 */
void RequantShift_parallel_s16_s8_NCHW(int16_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t HW, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_parallel_s32_s8_NCHW
 * layout           = NCHW
 * input data type  = 32-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = no
 * simd             = no
 */
void RequantShift_parallel_s32_s8_NCHW(int32_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t HW, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_unrolled_1x4_parallel_s32_s8_NCHW_rv32im
 * layout           = NCHW
 * input data type  = 32-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = yes (4 elements per iteration)
 * simd             = no
 */
void RequantShift_unrolled_1x4_parallel_s32_s8_NCHW_rv32im(
    int32_t *data_in, uint32_t size, int32_t *mul, int32_t *add,
    int8_t *data_out, int32_t log2D, uint32_t HW, int32_t input_offset,
    int32_t output_offset, bool rounding, uint32_t core_id,
    uint32_t numThreads);

#ifdef __XPULPIMG
/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_unrolled_1x2_parallel_s32_s8_NCHW_xpulpv2
 * layout           = NCHW
 * input data type  = 32-bit integer
 * output data type = 8-bit integer
 * multi-core       = yes
 * unrolling        = yes (4 elements per iteration)
 * simd             = no
 */
void RequantShift_unrolled_1x4_parallel_s32_s8_NCHW_xpulpv2(
    int32_t *data_in, uint32_t size, int32_t *mul, int32_t *add,
    int8_t *data_out, int32_t log2D, uint32_t HW, int32_t input_offset,
    int32_t output_offset, bool rounding, uint32_t core_id,
    uint32_t numThreads);

#endif //__XPULPIMG

// Mapper Functions
static inline void __attribute__((always_inline))
RequantShift_unrolled_1x4_parallel_s32_s8_NCHW(
    int32_t *data_in, uint32_t size, int32_t *mul, int32_t *add,
    int8_t *data_out, int32_t log2D, uint32_t HW, int32_t input_offset,
    int32_t output_offset, bool rounding, uint32_t core_id,
    uint32_t numThreads) {
#ifdef __XPULPIMG
  RequantShift_unrolled_1x4_parallel_s32_s8_NCHW_xpulpv2(
      data_in, size, mul, add, data_out, log2D, HW, input_offset, output_offset,
      rounding, core_id, numThreads);
#else
  RequantShift_unrolled_1x4_parallel_s32_s8_NCHW_rv32im(
      data_in, size, mul, add, data_out, log2D, HW, input_offset, output_offset,
      rounding, core_id, numThreads);
#endif
}

#endif //__DEEPLOY_MATH_REQUANTSHIFT_KERNEL_HEADER_
