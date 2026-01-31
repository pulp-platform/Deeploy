/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"
#include "Gemm.h"

void gemm_fp32_transB_opt(uint32_t M, uint32_t N, uint32_t K, float32_t *A,
                          uint32_t ldA, float32_t *B, uint32_t ldB,
                          float32_t *C, uint32_t ldC, float32_t *Y,
                          uint32_t BETA, uint32_t setup_SSR) {
  (void)setup_SSR;

  uint32_t compute_id = snrt_global_compute_core_idx();
  uint32_t A_offset = K * compute_id;
  uint32_t C_offset = N * compute_id;

  for (uint32_t m = 0; m < M; m++) {
    for (uint32_t n = 0; n < N; n++) {
      float32_t c;
      if (BETA) {
        c = C[C_offset + m * ldC + n];
      } else {
        c = 0.0f;
      }
      for (uint32_t k = 0; k < K; k++) {
        c += A[A_offset + m * ldA + k] * B[n * ldB + k];
      }
      Y[C_offset + m * ldC + n] = c;
    }
  }
}

void gemm_fp32_opt(uint32_t M, uint32_t N, uint32_t K, float32_t *A,
                   uint32_t ldA, float32_t *B, uint32_t ldB, float32_t *C,
                   uint32_t ldC, float32_t *Y, uint32_t BETA,
                   uint32_t setup_SSR) {
  (void)setup_SSR;

  uint32_t compute_id = snrt_global_compute_core_idx();
  uint32_t A_offset = K * compute_id;
  uint32_t C_offset = N * compute_id;

  for (uint32_t m = 0; m < M; m++) {
    for (uint32_t n = 0; n < N; n++) {
      float32_t c;
      if (BETA) {
        c = C[C_offset + m * ldC + n];
      } else {
        c = 0.0f;
      }
      for (uint32_t k = 0; k < K; k++) {
        c += A[A_offset + m * ldA + k] * B[k * ldB + n];
      }
      Y[C_offset + m * ldC + n] = c;
    }
  }
}
