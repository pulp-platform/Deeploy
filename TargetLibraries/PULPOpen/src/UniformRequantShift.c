/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

#define UNIFORM_REQUANT_SHIFT(IN, IN_T, OUT, OUT_T)                            \
  void UniformRequantShift_##IN##_##OUT(                                       \
      IN_T *data_in, int32_t size, int32_t mul, int32_t add, OUT_T *data_out,  \
      int32_t log2D, int32_t HW, int32_t input_offset, int32_t output_offset,  \
      OUT_T output_min, OUT_T output_max, bool rounding) {                     \
                                                                               \
    int32_t chunk =                                                            \
        (size >> LOG2(NUM_CORES)) + ((size & (NUM_CORES - 1)) != 0);           \
    int32_t start = MIN(chunk * (int32_t)pi_core_id(), size);                  \
    int32_t stop = MIN(start + chunk, size);                                   \
    if (start >= stop) {                                                       \
      return;                                                                  \
    }                                                                          \
                                                                               \
    int32_t rnd = (1 << (log2D - 1)) * rounding;                               \
    /* Compiler magic, don't remove the volatile keyword below */              \
    int32_t volatile half = (stop - start) >> 1;                               \
    int32_t intermediate;                                                      \
    IN_T reg_data_in_A = data_in[start];                                       \
    IN_T reg_data_in_B;                                                        \
                                                                               \
    for (int32_t i = start; i < start + half; i++) {                           \
      reg_data_in_B = data_in[half + i];                                       \
                                                                               \
      intermediate = (reg_data_in_A + input_offset) * mul + add;               \
      intermediate = ((intermediate + rnd) >> log2D) + output_offset;          \
      data_out[i] = (OUT_T)CLAMP(intermediate, output_min, output_max);        \
                                                                               \
      reg_data_in_A = data_in[i + 1];                                          \
                                                                               \
      intermediate = (reg_data_in_B + input_offset) * mul + add;               \
      intermediate = ((intermediate + rnd) >> log2D) + output_offset;          \
      data_out[half + i] = (OUT_T)CLAMP(intermediate, output_min, output_max); \
    }                                                                          \
                                                                               \
    if ((stop - start) & 1) {                                                  \
      intermediate = (data_in[stop - 1] + input_offset) * mul + add;           \
      intermediate = ((intermediate + rnd) >> log2D) + output_offset;          \
      data_out[stop - 1] = (OUT_T)CLAMP(intermediate, output_min, output_max); \
    }                                                                          \
  }

UNIFORM_REQUANT_SHIFT(s8, int8_t, s8, int8_t)
UNIFORM_REQUANT_SHIFT(u8, uint8_t, s8, int8_t)
UNIFORM_REQUANT_SHIFT(s16, int16_t, s8, int8_t)
UNIFORM_REQUANT_SHIFT(s32, int32_t, s8, int8_t)
