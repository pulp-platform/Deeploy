/* =====================================================================
 * Title:        RequantShift_s8.c
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

#include "DeeployMath.h"

void RequantShift_parallel_s8_s8_NHWC(int8_t *data_in, uint32_t size,
                                      int32_t *mul, int32_t *add,
                                      int8_t *data_out, int32_t log2D,
                                      uint32_t channels, int32_t input_offset,
                                      int32_t output_offset, int8_t output_min,
                                      int8_t output_max, bool rounding,
                                      uint32_t core_id, uint32_t numThreads) {
  int32_t intermediate;
  int8_t out;
  for (uint32_t i = core_id; i < size; i += numThreads) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_parallel_s16_s8_NHWC(int16_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t channels, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads) {
  int32_t intermediate;
  int8_t out;
  for (uint32_t i = core_id; i < size; i += numThreads) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_parallel_s32_s8_NHWC(int32_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t channels, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads) {
  int32_t intermediate;
  int8_t out;
  for (uint32_t i = core_id; i < size; i += numThreads) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_parallel_s8_s8_NCHW(int8_t *data_in, uint32_t size,
                                      int32_t *mul, int32_t *add,
                                      int8_t *data_out, int32_t log2D,
                                      uint32_t HW, int32_t input_offset,
                                      int32_t output_offset, int8_t output_min,
                                      int8_t output_max, bool rounding,
                                      uint32_t core_id, uint32_t numThreads) {
  int32_t intermediate;
  int8_t out;
  for (uint32_t i = core_id; i < size; i += numThreads) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_parallel_s16_s8_NCHW(int16_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t HW, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads) {
  int32_t intermediate;
  int8_t out;
  for (uint32_t i = core_id; i < size; i += numThreads) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_parallel_s32_s8_NCHW(int32_t *data_in, uint32_t size,
                                       int32_t *mul, int32_t *add,
                                       int8_t *data_out, int32_t log2D,
                                       uint32_t HW, int32_t input_offset,
                                       int32_t output_offset, int8_t output_min,
                                       int8_t output_max, bool rounding,
                                       uint32_t core_id, uint32_t numThreads) {
  int32_t intermediate;
  int8_t out;
  for (uint32_t i = core_id; i < size; i += numThreads) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_unrolled_1x4_parallel_s32_s8_NCHW_rv32im(
    int32_t *data_in, uint32_t size, int32_t *mul, int32_t *add,
    int8_t *data_out, int32_t log2D, uint32_t HW, int32_t input_offset,
    int32_t output_offset, bool rounding, uint32_t core_id,
    uint32_t numThreads) {

  const int32_t round_bias = ((1 << (log2D - 1))) * rounding;

  for (uint32_t i = core_id; i < size / 4; i += numThreads) {
    int32_t shifted0, shifted1, shifted2, shifted3;

    shifted0 = (((data_in[i * 4 + 0] + input_offset) * mul[(i * 4 + 0) / HW] +
                 add[(i * 4 + 0) / HW] + round_bias) >>
                log2D) +
               output_offset;
    shifted1 = (((data_in[i * 4 + 1] + input_offset) * mul[(i * 4 + 1) / HW] +
                 add[(i * 4 + 1) / HW] + round_bias) >>
                log2D) +
               output_offset;
    shifted2 = (((data_in[i * 4 + 2] + input_offset) * mul[(i * 4 + 2) / HW] +
                 add[(i * 4 + 2) / HW] + round_bias) >>
                log2D) +
               output_offset;
    shifted3 = (((data_in[i * 4 + 3] + input_offset) * mul[(i * 4 + 3) / HW] +
                 add[(i * 4 + 3) / HW] + round_bias) >>
                log2D) +
               output_offset;

    data_out[i * 4 + 0] = (int8_t)(CLAMP(shifted0, -128, 127));
    data_out[i * 4 + 1] = (int8_t)(CLAMP(shifted1, -128, 127));
    data_out[i * 4 + 2] = (int8_t)(CLAMP(shifted2, -128, 127));
    data_out[i * 4 + 3] = (int8_t)(CLAMP(shifted3, -128, 127));
  }
}

#ifdef __XPULPIMG
void RequantShift_unrolled_1x4_parallel_s32_s8_NCHW_xpulpv2(
    int32_t *data_in, uint32_t size, int32_t *mul, int32_t *add,
    int8_t *data_out, int32_t log2D, uint32_t HW, int32_t input_offset,
    int32_t output_offset, bool rounding, uint32_t core_id,
    uint32_t numThreads) {

  const int32_t round_bias = ((1 << (log2D - 1))) * rounding;

  for (uint32_t i = core_id; i < size / 4; i += numThreads) {
    int32_t shifted0, shifted1, shifted2, shifted3;

    shifted0 = (((data_in[i * 4 + 0] + input_offset) * mul[(i * 4 + 0) / HW] +
                 add[(i * 4 + 0) / HW] + round_bias) >>
                log2D) +
               output_offset;
    shifted1 = (((data_in[i * 4 + 1] + input_offset) * mul[(i * 4 + 1) / HW] +
                 add[(i * 4 + 1) / HW] + round_bias) >>
                log2D) +
               output_offset;
    shifted2 = (((data_in[i * 4 + 2] + input_offset) * mul[(i * 4 + 2) / HW] +
                 add[(i * 4 + 2) / HW] + round_bias) >>
                log2D) +
               output_offset;
    shifted3 = (((data_in[i * 4 + 3] + input_offset) * mul[(i * 4 + 3) / HW] +
                 add[(i * 4 + 3) / HW] + round_bias) >>
                log2D) +
               output_offset;

    data_out[i * 4 + 0] = (int8_t)(__CLIP(shifted0, 7));
    data_out[i * 4 + 1] = (int8_t)(__CLIP(shifted1, 7));
    data_out[i * 4 + 2] = (int8_t)(__CLIP(shifted2, 7));
    data_out[i * 4 + 3] = (int8_t)(__CLIP(shifted3, 7));
  }
}
#endif //__XPULPIMG