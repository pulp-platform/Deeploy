/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void ReduceLogSumExp_fp32_fp32(float32_t *input, float32_t *output,
                               uint32_t outer_size, uint32_t axis_length,
                               uint32_t inner_size) {

  for (uint32_t outer_idx = 0; outer_idx < outer_size; ++outer_idx) {
    for (uint32_t inner_idx = 0; inner_idx < inner_size; ++inner_idx) {
      float32_t max_val = -INFINITY;

      for (uint32_t axis_idx = 0; axis_idx < axis_length; ++axis_idx) {
        uint32_t input_idx =
            (outer_idx * axis_length + axis_idx) * inner_size + inner_idx;
        if (input[input_idx] > max_val) {
          max_val = input[input_idx];
        }
      }

      float32_t sum_exp = 0.0f;
      for (uint32_t axis_idx = 0; axis_idx < axis_length; ++axis_idx) {
        uint32_t input_idx =
            (outer_idx * axis_length + axis_idx) * inner_size + inner_idx;
        sum_exp += expf(input[input_idx] - max_val);
      }

      output[outer_idx * inner_size + inner_idx] = logf(sum_exp) + max_val;
    }
  }
}
