/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"
#include <math.h>

void RMSNorm_fp32(float32_t *data_in, float32_t *weight, float32_t *data_out,
                  uint32_t size, uint32_t lastDimLength, float32_t eps) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t num_vectors = size / lastDimLength;

  // Parallelize across vectors (batch * sequence dimension)
  uint32_t vectors_per_core = num_vectors / numThreads;
  uint32_t remainder = num_vectors % numThreads;

  uint32_t start_vec, num_vecs;
  if (core_id < remainder) {
    num_vecs = vectors_per_core + 1;
    start_vec = core_id * num_vecs;
  } else {
    num_vecs = vectors_per_core;
    start_vec = core_id * vectors_per_core + remainder;
  }

  for (uint32_t v = start_vec; v < start_vec + num_vecs; v++) {
    float32_t *in_ptr = data_in + v * lastDimLength;
    float32_t *out_ptr = data_out + v * lastDimLength;

    // Compute sum of squares
    float32_t sum_sq = 0.0f;
    for (uint32_t i = 0; i < lastDimLength; i++) {
      sum_sq += in_ptr[i] * in_ptr[i];
    }

    // Compute RMS with epsilon
    float32_t rms = sqrtf(sum_sq / (float32_t)lastDimLength + eps);
    float32_t inv_rms = 1.0f / rms;

    // Apply normalization and weight
    for (uint32_t i = 0; i < lastDimLength; i++) {
      out_ptr[i] = in_ptr[i] * inv_rms * weight[i];
    }
  }
}

/*
 * RMSNorm (FP32) with SSR + FREP on the sum-of-squares reduction.
 *
 * Only the reduction phase uses SSR/FREP: it streams the input vector as v2f32
 * pairs through SSR DM0 (read) and accumulates squares into a v2f32 register
 * via FREP vfmac.s. This is the same register-accumulate pattern as the SSR
 * MatMul — it never uses an SSR write stream (DM2), so it is safe in the tiled
 * model flow (the DM2 write FIFO does not reliably drain there; see notes).
 *
 * The output scale (out = in * inv_rms * weight) writes element-wise, so it
 * stays a plain loop with normal stores (no DM2). rsqrt is a per-vector scalar.
 *
 * SSR reduction needs an 8-byte-aligned base, which holds when lastDimLength is
 * even (in_ptr = base + v*L); for odd lastDimLength it falls back to scalar.
 *
 * NOTE: Snitch SSR can only stream from cluster TCDM/L1 (tiled flow).
 */
void RMSNorm_fp32_ssr_frep(float32_t *data_in, float32_t *weight,
                           float32_t *data_out, uint32_t size,
                           uint32_t lastDimLength, float32_t eps) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t num_vectors = size / lastDimLength;

  uint32_t vectors_per_core = num_vectors / numThreads;
  uint32_t remainder = num_vectors % numThreads;

  uint32_t start_vec, num_vecs;
  if (core_id < remainder) {
    num_vecs = vectors_per_core + 1;
    start_vec = core_id * num_vecs;
  } else {
    num_vecs = vectors_per_core;
    start_vec = core_id * vectors_per_core + remainder;
  }

  uint32_t L = lastDimLength;
  uint32_t npairs = L / 2;
  int use_ssr = ((L & 1u) == 0) && (npairs > 0);

  for (uint32_t v = start_vec; v < start_vec + num_vecs; v++) {
    float32_t *in_ptr = data_in + v * L;
    float32_t *out_ptr = data_out + v * L;

    // --- Sum of squares ---
    float32_t sum_sq = 0.0f;
    if (use_ssr) {
      // SSR read v2f32 + FREP register accumulate (no DM2 write stream).
      v2f32 acc = {0.0f, 0.0f};
      snrt_ssr_loop_1d(SNRT_SSR_DM0, npairs, sizeof(float32_t) * 2);
      // Reset the repeat count: a prior SSR kernel (e.g. matmul) may have left
      // snrt_ssr_repeat(DM0, >1) set; it persists and would re-read each element.
      snrt_ssr_repeat(SNRT_SSR_DM0, 1);
      snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_1D, in_ptr);
      snrt_ssr_enable();
      asm volatile("frep.o %[n_frep], 1, 0, 0 \n"
                   "vfmac.s %[acc], ft0, ft0 \n"
                   : [acc] "+f"(acc)
                   : [n_frep] "r"(npairs - 1)
                   : "ft0", "memory");
      snrt_fpu_fence();
      snrt_ssr_disable();
      sum_sq = acc[0] + acc[1];
    } else {
      for (uint32_t i = 0; i < L; i++) {
        sum_sq += in_ptr[i] * in_ptr[i];
      }
    }

    // --- RMS + reciprocal (scalar, once per vector) ---
    float32_t rms = sqrtf(sum_sq / (float32_t)L + eps);
    float32_t inv_rms = 1.0f / rms;

    // --- Scale + weight (element-wise, normal stores; no DM2) ---
    for (uint32_t i = 0; i < L; i++) {
      out_ptr[i] = in_ptr[i] * inv_rms * weight[i];
    }
  }
}
