/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "pmsis.h"

#include "DeeployPULPMath.h"

#include <math.h>

/*
 * Training-mode Batch Normalization forward pass (BatchNormInternal).
 *
 * Computes per-channel batch statistics over (N, H, W) and normalizes.
 * Saves batch mean and 1/sqrt(var+eps) into stash buffers for the backward pass.
 * Running statistics are NOT updated (their outputs have no consumers in the graph).
 *
 * Layout: NCHW — element [n, c, h, w] lives at offset (n*C + c)*N_hw + h*W_in + w.
 *
 * Parallelism: channels are split evenly across cores.
 */
void PULP_BatchNormInternal_fp32(const float32_t *X, const float32_t *gamma,
                                 const float32_t *beta, const float32_t *running_mean,
                                 const float32_t *running_var, float32_t *Y,
                                 float32_t *saved_mean, float32_t *saved_inv_std,
                                 uint32_t N, uint32_t C, uint32_t H_in, uint32_t W_in,
                                 float32_t epsilon, float32_t momentum) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;
  uint32_t N_total = N * N_hw;
  float32_t inv_N = 1.0f / (float32_t)N_total;

  /* Split channels across cores */
  int32_t chunk = ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    /* ── Compute batch mean ─────────────────────────────────────────────── */
    float32_t mean = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        mean += x_nc[hw];
      }
    }
    mean *= inv_N;
    saved_mean[c] = mean;

    /* ── Compute batch variance (unbiased=False) ────────────────────────── */
    float32_t var = 0.0f;
    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        float32_t diff = x_nc[hw] - mean;
        var += diff * diff;
      }
    }
    var *= inv_N;

    float32_t inv_std = 1.0f / sqrtf(var + epsilon);
    saved_inv_std[c] = inv_std;

    float32_t g = gamma[c];
    float32_t b = beta[c];

    /* ── Normalize and apply affine transform ───────────────────────────── */
    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      float32_t *y_nc = Y + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        y_nc[hw] = (x_nc[hw] - mean) * inv_std * g + b;
      }
    }
  }
}

/*
 * Batch Normalization backward pass.
 *
 * Uses saved_mean / saved_inv_std stash from the forward pass to compute:
 *   dbeta[c]    = sum_{n,h,w} dY[n,c,h,w]
 *   dgamma[c]   = sum_{n,h,w} dY[n,c,h,w] * x_hat[n,c,h,w]
 *   dX[n,c,h,w] = inv_std / N_total * (N_total * dY[n,c,h,w] * gamma[c]
 *                                       - dbeta[c]
 *                                       - x_hat[n,c,h,w] * dgamma[c])
 *
 * Parallelism: channels are split evenly across cores.
 * Each core independently computes its channel slice of dgamma, dbeta, dX.
 */
void PULP_BatchNormGrad_fp32(const float32_t *dY, const float32_t *X,
                              const float32_t *gamma, const float32_t *saved_mean,
                              const float32_t *saved_inv_std, float32_t *dX,
                              float32_t *dgamma, float32_t *dbeta, uint32_t N, uint32_t C,
                              uint32_t H_in, uint32_t W_in, float32_t epsilon) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint32_t N_hw = H_in * W_in;
  uint32_t N_total = N * N_hw;
  float32_t inv_N = 1.0f / (float32_t)N_total;

  /* Split channels across cores */
  int32_t chunk = ((int32_t)C >> log2Core) + (((int32_t)C & (NUM_CORES - 1)) != 0);
  int32_t c_start = MIN(chunk * core_id, (int32_t)C);
  int32_t c_end = MIN(c_start + chunk, (int32_t)C);

  for (int32_t c = c_start; c < c_end; c++) {
    float32_t mean = saved_mean[c];
    float32_t inv_std = saved_inv_std[c];
    float32_t g = gamma[c];

    /* ── First pass: accumulate dbeta and dgamma ─────────────────────────── */
    float32_t sum_dbeta = 0.0f;
    float32_t sum_dgamma = 0.0f;

    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      const float32_t *dy_nc = dY + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        float32_t x_hat = (x_nc[hw] - mean) * inv_std;
        sum_dbeta += dy_nc[hw];
        sum_dgamma += dy_nc[hw] * x_hat;
      }
    }
    dgamma[c] = sum_dgamma;
    dbeta[c] = sum_dbeta;

    /* ── Second pass: compute dX ─────────────────────────────────────────── */
    float32_t scale = inv_std * inv_N;

    for (uint32_t n = 0; n < N; n++) {
      const float32_t *x_nc = X + (n * C + c) * N_hw;
      const float32_t *dy_nc = dY + (n * C + c) * N_hw;
      float32_t *dx_nc = dX + (n * C + c) * N_hw;
      for (uint32_t hw = 0; hw < N_hw; hw++) {
        float32_t x_hat = (x_nc[hw] - mean) * inv_std;
        float32_t dx_hat = dy_nc[hw] * g;
        dx_nc[hw] = scale * ((float32_t)N_total * dx_hat - sum_dbeta - x_hat * sum_dgamma);
      }
    }
  }
}
