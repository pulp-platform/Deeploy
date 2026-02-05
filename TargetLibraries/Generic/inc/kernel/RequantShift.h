/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_REQUANTSHIFT_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_REQUANTSHIFT_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This file implements the requantization kernel for several data widths
 * in multiple different ways.
 */

/******************************************************************************/
/*                         Requantization to 8bit                             */
/******************************************************************************/

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_s8_s8_NHWC
 * layout           = NHWC
 * input data type  = 8-bit integer
 * output data type = 8-bit integer
 * unrolling        = no
 * simd             = no
 */
void RequantShift_s8_s8_NHWC(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_s16_s8_NHWC
 * layout           = NHWC
 * input data type  = 16-bit integer
 * output data type = 8-bit integer
 * unrolling        = no
 * simd             = no
 */
void RequantShift_s16_s8_NHWC(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_s32_s8_NHWC
 * layout           = NHWC
 * input data type  = 32-bit integer
 * output data type = 8-bit integer
 * unrolling        = no
 * simd             = no
 */
void RequantShift_s32_s8_NHWC(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_s16_s8_NCHW
 * layout           = NCHW
 * input data type  = 16-bit integer
 * output data type = 8-bit integer
 * unrolling        = no
 * simd             = no
 */
void RequantShift_s8_s8_NCHW(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_s16_s8_NCHW
 * layout           = NCHW
 * input data type  = 16-bit integer
 * output data type = 8-bit integer
 * unrolling        = no
 * simd             = no
 */
void RequantShift_s16_s8_NCHW(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

/*
 * Re-quantization and Shift  ----------------------------------
 * kernel           = RequantShift_s32_s8_NCHW
 * layout           = NCHW
 * input data type  = 32-bit integer
 * output data type = 8-bit integer
 * unrolling        = no
 * simd             = no
 */
void RequantShift_s32_s8_NCHW(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding);

#endif //__DEEPLOY_BASIC_MATH_REQUANTSHIFT_KERNEL_HEADER_
