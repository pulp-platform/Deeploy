/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

void Softmax_fp32(float32_t *input, float32_t *output, int32_t ldI,
                  int32_t batch_offset, int32_t batch_size, int32_t seq_len,
                  int32_t input_samples) {

  float32_t max_core = 0.0; // max value of the current core
  float32_t sum = 0.0;      // sum of the exp values of the current core
  int32_t compute_id = snrt_global_compute_core_idx();
  int32_t row_offset = compute_id * input_samples;
  for (int32_t b = 0; b < batch_size; b++) {
    for (int32_t s = 0; s < seq_len; s++) {
      max_core = -INFINITY;
      sum = 0.0;
      for (int32_t i = 0; i < input_samples; i++) {
        if (input[row_offset + b * batch_offset + s * ldI + i] > max_core) {
          max_core = input[row_offset + b * batch_offset + s * ldI + i];
        }
      }
      // compute the shifted value of the current row
      for (int32_t i = 0; i < input_samples; i++) {
        output[row_offset + b * batch_offset + s * ldI + i] =
            expf(input[row_offset + b * batch_offset + s * ldI + i] - max_core);
        sum += output[row_offset + b * batch_offset + s * ldI + i];
      }
      // compute the softmax value of the current row
      for (int32_t i = 0; i < input_samples; i++) {
        output[row_offset + b * batch_offset + s * ldI + i] /= sum;
      }
    }
  }
}
