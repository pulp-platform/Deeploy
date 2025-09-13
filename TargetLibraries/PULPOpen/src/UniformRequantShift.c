/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void UniformRequantShift_s8_s8(int8_t *data_in, int32_t size, int32_t mul,
                               int32_t add, int8_t *data_out, int32_t log2D,
                               int32_t HW, int32_t input_offset,
                               int32_t output_offset, int8_t output_min,
                               int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, size);
  int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  int8_t out;
  int8_t reg_data_in_A;
  int8_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}

void UniformRequantShift_u8_s8(uint8_t *data_in, int32_t size, int32_t mul,
                               int32_t add, int8_t *data_out, int32_t log2D,
                               int32_t HW, int32_t input_offset,
                               int32_t output_offset, int8_t output_min,
                               int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, size);
  int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  int8_t out;
  uint8_t reg_data_in_A;
  uint8_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}

void UniformRequantShift_s16_s8(int16_t *data_in, int32_t size, int32_t mul,
                                int32_t add, int8_t *data_out, int32_t log2D,
                                int32_t HW, int32_t input_offset,
                                int32_t output_offset, int8_t output_min,
                                int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, size);
  int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  int8_t out;
  int16_t reg_data_in_A;
  int16_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}

void UniformRequantShift_s32_s8(int32_t *data_in, int32_t size, int32_t mul,
                                int32_t add, int8_t *data_out, int32_t log2D,
                                int32_t HW, int32_t input_offset,
                                int32_t output_offset, int8_t output_min,
                                int8_t output_max, bool rounding) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, size);
  int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);

  // JUNGVI: Compiler magic, don't remove the volatile keyword below
  int32_t volatile halfChunkSize = chunk >> 1;
  int32_t intermediate;
  int8_t out;
  int32_t reg_data_in_A;
  int32_t reg_data_in_B;

  // Load step 0
  reg_data_in_A = data_in[chunk_start];

  for (int i = chunk_start; i < chunk_start + halfChunkSize; i++) {

    // Load step halfChunkSize + i
    reg_data_in_B = data_in[halfChunkSize + i];

    // Compute i
    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[i] = out;

    // Load step i + 1
    reg_data_in_A = data_in[i + 1];

    // Compute step halfChunkSize + i
    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[halfChunkSize + i] = out;
  }

  // Leftover computation
  if ((chunk_stop - chunk_start) % 2) {

    reg_data_in_B = data_in[chunk_stop - 1];
    reg_data_in_A = data_in[chunk_stop];

    intermediate = (reg_data_in_B + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop - 1] = out;

    intermediate = (reg_data_in_A + input_offset) * mul + add;
    intermediate = ((intermediate + ((1 << (log2D - 1))) * rounding) >> log2D) +
                   output_offset;
    out = (int8_t)CLAMP(intermediate, output_min, output_max);
    data_out[chunk_stop] = out;
  }
}