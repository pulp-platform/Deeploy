/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void RequantShift_u8_s8_NHWC(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u16_s8_NHWC(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u32_s8_NHWC(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u8_s8_NCHW(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, int8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, int8_t output_min,
                             int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u16_s8_NCHW(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u32_s8_NCHW(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, int8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, int8_t output_min,
                              int8_t output_max, bool rounding) {
  int32_t intermediate;
  int8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u8_u8_NHWC(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u16_u8_NHWC(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u32_u8_NHWC(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u8_u8_NCHW(uint8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u16_u8_NCHW(uint16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_u32_u8_NCHW(uint32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP((uint32_t)intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s8_u8_NHWC(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t channels, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s16_u8_NHWC(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s32_u8_NHWC(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t channels, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate = ((int32_t)data_in[i] + input_offset) * mul[i % channels] +
                   add[i % channels];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s8_u8_NCHW(int8_t *data_in, int32_t size, int32_t *mul,
                             int32_t *add, uint8_t *data_out, int32_t log2D,
                             int32_t HW, int32_t input_offset,
                             int32_t output_offset, uint8_t output_min,
                             uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s16_u8_NCHW(int16_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}

void RequantShift_s32_u8_NCHW(int32_t *data_in, int32_t size, int32_t *mul,
                              int32_t *add, uint8_t *data_out, int32_t log2D,
                              int32_t HW, int32_t input_offset,
                              int32_t output_offset, uint8_t output_min,
                              uint8_t output_max, bool rounding) {
  int32_t intermediate;
  uint8_t out;
  for (int i = 0; i < size; i++) {
    intermediate =
        ((int32_t)data_in[i] + input_offset) * mul[i / HW] + add[i / HW];
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (uint8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;
  }
}
