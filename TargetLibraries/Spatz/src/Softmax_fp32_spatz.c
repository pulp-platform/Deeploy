/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

float32_t myexpf(float32_t x){
  const float32_t inv_ln2 = 1.4426950409f;
  const float32_t ln2 = 0.6931471806f;

  // Range reduction: x = k * ln(2) + r, with r kept small so the polynomial is accurate.
  float32_t scaled = x * inv_ln2;
  int32_t k = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
  float32_t r = x - ((float32_t)k * ln2);

  float32_t r2 = r * r;
  float32_t r3 = r2 * r;
  float32_t r4 = r3 * r;
  float32_t r5 = r4 * r;
  float32_t r6 = r5 * r;
  float32_t r7 = r6 * r;

  float32_t poly = 1.0f + r + (r2 * 0.5f) + (r3 * 0.1666666667f) + (r4 * 0.0416666667f) + (r5 * 0.0083333333f) + (r6 * 0.0013888889f) + (r7 * 0.0001984127f);

  return ldexpf(poly, k);
}

// inverse funciton that doesnt use fdiv.s
float32_t myinv(float32_t x){
    uint32_t i = *(uint32_t*)&x;
    i = 0x7EEEEEEE - i; 
    float y = *(float*)&i;

    // Newton-Raphson steps (Multiplication only!)
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y);
    y = y * (2.0f - x * y); 
    
    return y;
}

void Spatz_Softmax_fp32_fp32(float32_t *input, float32_t *output, int32_t size, int32_t last_dim_length) {
  const unsigned int cid = snrt_cluster_core_idx();
  int32_t batch_size = size / last_dim_length;
  // divide in two cores
  unsigned int items_per_core = (batch_size + 1) / 2;

  unsigned int b_start, b_end;

  if (cid == 0) {
      b_start = 0;
      b_end   = items_per_core;
  } else {
      b_start = items_per_core;
      // Core 1 always ends at the total batch size
      b_end   = batch_size;
  }
  for (int b = b_start; b < b_end; b++) {
    float32_t max_val = -inf;
    float sum = 0.0f;

    for (int i = 0; i < last_dim_length; i++) {
      if (input[b * last_dim_length + i] > max_val) {
        max_val = input[b * last_dim_length + i];
      }
    }

    for (int i = 0; i < last_dim_length; i++) {
      float32_t exp_val = input[b * last_dim_length + i] - max_val;
      output[b * last_dim_length + i] = myexpf(exp_val);
      sum += output[b * last_dim_length + i];
    }

    float32_t sum_1 = myinv(sum);
    for (int i = 0; i < last_dim_length; i++) {
      output[b * last_dim_length + i] = output[b * last_dim_length + i] * sum_1;
    }
  }
}
