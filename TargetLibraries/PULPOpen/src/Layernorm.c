/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

#include <math.h>

/*
 * Forward pass: LayerNorm with stash output
 *
 * Normalizes along the last dimension (axis=-1).
 * Parallelized across sequence positions: each core processes a chunk of
 * sequences. In addition to the normalized output (data_out), writes the
 * per-sequence mean and 1/sqrt(var+eps) into mean_out and inv_std_dev_out
 * so the backward pass can reuse them without recomputation.
 */
void PULP_Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias,
                              float32_t *mean_out, float32_t *inv_std_dev_out,
                              uint32_t size, uint32_t lastDimLength,
                              float32_t epsilon) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  int32_t seq_length = size / lastDimLength;
  int32_t chunk =
      (seq_length >> log2Core) + ((seq_length & (NUM_CORES - 1)) != 0);
  int32_t start_seq = MIN(chunk * core_id, seq_length);
  int32_t end_seq = MIN(start_seq + chunk, seq_length);

  int32_t elem_start = start_seq * lastDimLength;
  int32_t elem_end = end_seq * lastDimLength;

  float32_t *local_data_in = data_in + elem_start;
  float32_t *local_data_out = data_out + elem_start;
  int32_t local_size = elem_end - elem_start;
  int32_t local_seq_count = local_size / lastDimLength;

  for (int32_t i = 0; i < local_seq_count; i++) {
    float32_t *row_in = local_data_in + i * lastDimLength;
    float32_t *row_out = local_data_out + i * lastDimLength;

    /* Compute mean */
    float32_t mean = 0.0f;
    for (int32_t j = 0; j < lastDimLength; j++) {
      mean += row_in[j];
    }
    mean /= (float32_t)lastDimLength;

    /* Compute variance */
    float32_t var = 0.0f;
    for (int32_t j = 0; j < lastDimLength; j++) {
      float32_t diff = row_in[j] - mean;
      var += diff * diff;
    }
    var /= (float32_t)lastDimLength;

    float32_t isd = 1.0f / sqrtf(var + epsilon);

    /* Write stash (indexed by global sequence position) */
    mean_out[start_seq + i] = mean;
    inv_std_dev_out[start_seq + i] = isd;

    /* Compute normalized output */
    for (int32_t j = 0; j < lastDimLength; j++) {
      row_out[j] = (row_in[j] - mean) * isd * scale[j] + bias[j];
    }
  }
}

/*
 * Backward pass: compute dX for a chunk of sequences.
 *
 * Uses the pre-computed mean and inv_std_dev stash from the forward pass.
 * Called per-core with each core's local chunk.
 *
 * Math (standard LayerNorm backward with axis=-1):
 *   x_hat[i] = (x[i] - mean) * isd
 *   mean_dy   = sum(dy) / N
 *   mean_dy_xhat = sum(dy * x_hat) / N
 *   dx[i] = gamma[i] * isd * (dy[i] - mean_dy - x_hat[i] * mean_dy_xhat)
 */
void PULP_LayernormGrad_fp32_fp32(const float32_t *dy, const float32_t *x,
                             const float32_t *mean, const float32_t *inv_std_dev,
                             float32_t *dx, const float32_t *gamma,
                             uint32_t elem_count, uint32_t lastDimLength) {

  uint32_t seq_count = elem_count / lastDimLength;

  for (uint32_t s = 0; s < seq_count; s++) {
    const float32_t *dy_s = dy + s * lastDimLength;
    const float32_t *x_s  = x  + s * lastDimLength;
    float32_t *dx_s        = dx + s * lastDimLength;
    float32_t m   = mean[s];
    float32_t isd = inv_std_dev[s];

    /* Accumulate sum_dy and sum_dy_xhat */
    float32_t sum_dy      = 0.0f;
    float32_t sum_dy_xhat = 0.0f;
    for (uint32_t i = 0; i < lastDimLength; i++) {
      float32_t x_hat_i = (x_s[i] - m) * isd;
      sum_dy      += dy_s[i];
      sum_dy_xhat += dy_s[i] * x_hat_i;
    }
    float32_t mean_dy      = sum_dy      / (float32_t)lastDimLength;
    float32_t mean_dy_xhat = sum_dy_xhat / (float32_t)lastDimLength;

    /* Compute dX */
    for (uint32_t i = 0; i < lastDimLength; i++) {
      float32_t x_hat_i = (x_s[i] - m) * isd;
      dx_s[i] = gamma[i] * isd * (dy_s[i] - mean_dy - x_hat_i * mean_dy_xhat);
    }
  }
}

/*
 * Backward pass: compute dscale (dgamma) and dbias (dbeta) over all sequences.
 *
 * Called from core 0 only. Uses pre-computed mean and inv_std_dev stash.
 *
 * Math:
 *   dgamma[i] = sum_s( dy[s,i] * (x[s,i] - mean[s]) * isd[s] )
 *   dbeta[i]  = sum_s( dy[s,i] )
 */
void PULP_LayernormGradParam_fp32_fp32(const float32_t *dy, const float32_t *x,
                                  const float32_t *mean, const float32_t *inv_std_dev,
                                  float32_t *dgamma, float32_t *dbeta,
                                  uint32_t size, uint32_t lastDimLength) {

  uint32_t seq_length = size / lastDimLength;

  /* Initialize output gradients */
  for (uint32_t i = 0; i < lastDimLength; i++) {
    dgamma[i] = 0.0f;
    dbeta[i]  = 0.0f;
  }

  for (uint32_t s = 0; s < seq_length; s++) {
    const float32_t *dy_s = dy + s * lastDimLength;
    const float32_t *x_s  = x  + s * lastDimLength;
    float32_t m   = mean[s];
    float32_t isd = inv_std_dev[s];

    for (uint32_t i = 0; i < lastDimLength; i++) {
      float32_t x_hat_i = (x_s[i] - m) * isd;
      dgamma[i] += dy_s[i] * x_hat_i;
      dbeta[i]  += dy_s[i];
    }
  }
}
