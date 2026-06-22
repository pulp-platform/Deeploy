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

// Type-punning union to safely manipulate IEEE 754 float bits without breaking strict aliasing rules
union float_bits {
  float f;
  uint32_t i;
};

float expf_nodiv_reduced(float x) {
  // Mathematical constants
  const float LN2_HI = 0.69314575195f;     // High bits of ln(2)
  const float LN2_LO = 1.4286068203e-6f;   // Low bits of ln(2) for quasi-double precision reduction
  const float INV_LN2 = 1.4426950408f;     // log2(e) = 1/ln(2)

  // Bound limits to prevent float overflow/underflow
  if (x > 88.722839f)  x = 88.722839f;
  if (x < -87.336544f) return 0.0f;

  // 1. Argument Reduction: Find integer k closest to x / ln(2)
  // We cast to integer to perform a fast round-to-nearest operation
  // int32_t k = (int32_t)(x * INV_LN2 + (x >= 0.0f ? 0.5f : -0.5f));
  float sign_offset = __builtin_copysignf(0.5f, x);
  int32_t k = (int32_t)(x * INV_LN2 + sign_offset);

  // Compute residual r = x - k * ln(2) using Cody-Waite reduction to minimize loss of significance
  float r = x - ((float)k * LN2_HI) - ((float)k * LN2_LO);

  // 2. Taylor Polynomial Approximation of e^r (Horner's Method)
  // Range of r is strictly bounded within [-0.34657, 0.34657]
  // Coefficients are 1, 1, 1/2, 1/6, 1/24, 1/120
  float poly = 1.0f + r * (1.0f + r * (0.5f + r * (0.166666671f + r * (0.041666664f + r * 0.008333333f))));

  // 3. Reconstruction: Generate 2^k via IEEE 754 bit-manipulation
  // The exponent field is bits [30:23] with a bias of 127
  int32_t biased_exp = k + 127;
  
  union float_bits two_to_k;
  two_to_k.i = ((uint32_t)biased_exp << 23); // Shift biased integer into the float exponent slot

  // e^x = e^r * 2^k
  return poly * two_to_k.f;
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
  // two cores divided on the vector lenght
  if (size == last_dim_length){
    static float32_t maxval[1];
    if (cid==0){
      float32_t max_val = -inf;

      for (int i = 0; i < last_dim_length; i++) {
        if (input[i] > max_val) { max_val = input[i]; }
      }
      maxval[0] = max_val;
    }

    snrt_cluster_hw_barrier();

    static float32_t partial_sum[2];
    float32_t exp_val = 0.0f;

    if (cid==0){
      float32_t sum_core0 = 0.0f;
      for (int i = 0; i < last_dim_length/2; i++) {
        exp_val = expf_nodiv_reduced(input[i] - maxval[0]);
        output[i] = exp_val;
        sum_core0 += exp_val;
      }
      partial_sum[0] = sum_core0;
    } else {
      float32_t sum_core1 = 0.0f;
      for (int i = last_dim_length/2; i < last_dim_length; i++) {
        exp_val = expf_nodiv_reduced(input[i] - maxval[0]);
        output[i] = exp_val;
        sum_core1 += exp_val;
      }
      partial_sum[1] = sum_core1;
    }

    snrt_cluster_hw_barrier();
    float32_t one_over_sum= 0.0f;

    if (cid == 0){ one_over_sum = myinv(partial_sum[0] + partial_sum[1]); }
    snrt_cluster_hw_barrier();
    if (cid == 0){ for (int i = 0; i < last_dim_length; i++) { output[i] *= one_over_sum; } }
    snrt_cluster_hw_barrier();
    return;

  } else {
    // divide worload betw cores in batches
    int32_t batch_size = size / last_dim_length;
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
        output[b * last_dim_length + i] = expf_nodiv_reduced(exp_val);
        sum += output[b * last_dim_length + i];
      }

      float32_t sum_1 = myinv(sum);
      for (int i = 0; i < last_dim_length; i++) {
        output[b * last_dim_length + i] = output[b * last_dim_length + i] * sum_1;
      }
    }
  }
}
