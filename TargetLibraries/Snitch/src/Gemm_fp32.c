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

  uint32_t compute_id = snrt_global_compute_core_idx();
  uint32_t A_offset = K * compute_id;
  uint32_t C_offset = N * compute_id;

  // Unrolling factor of most inner loop.
  // Should be at least as high as the FMA delay
  // for maximum utilization
  const uint32_t unroll = 8;

  // SSR strides and bounds only have to be configured
  // once in the beginning
  if (setup_SSR) {
    // First matrix is not stored in transposed format
    const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M};
    const uint32_t ssr0_i[4] = {0, sizeof(float32_t), 0,
                                sizeof(float32_t) * ldA};

    // Second matrix is stored in transposed format
    const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M};
    const uint32_t ssr1_i[4] = {sizeof(float32_t) * K, sizeof(float32_t),
                                sizeof(float32_t) * K * unroll, 0};

    snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3], ssr0_i[1],
                     ssr0_i[2], ssr0_i[3]);

    snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2], ssr1_b[3],
                     ssr1_i[0], ssr1_i[1], ssr1_i[2], ssr1_i[3]);
  }

  // SSR start address need to be configured each time

  snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, &A[A_offset]);
  snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, B);
  snrt_ssr_enable();

  // check dimensions and values of a and b

  // Kernel progresses by 1 values each step
  // const uint32_t n_frep = K - 1;
  for (uint32_t m = 0; m < M; m++) {
    uint32_t n = 0;
    for (uint32_t n0 = 0; n0 < N / unroll; n0++) {
      float c[unroll];

      // Load intermediate result
      if (BETA) {
        c[0] = C[C_offset + m * ldC + n + 0];
        c[1] = C[C_offset + m * ldC + n + 1];
        c[2] = C[C_offset + m * ldC + n + 2];
        c[3] = C[C_offset + m * ldC + n + 3];
        c[4] = C[C_offset + m * ldC + n + 4];
        c[5] = C[C_offset + m * ldC + n + 5];
        c[6] = C[C_offset + m * ldC + n + 6];
        c[7] = C[C_offset + m * ldC + n + 7];
      } else {
        c[0] = 0.0;
        c[1] = 0.0;
        c[2] = 0.0;
        c[3] = 0.0;
        c[4] = 0.0;
        c[5] = 0.0;
        c[6] = 0.0;
        c[7] = 0.0;
      }

      asm volatile(
          "frep.o %[n_frep], 8, 0, 0 \n"
          "fmadd.s %[c0], ft0, ft1, %[c0] \n"
          "fmadd.s %[c1], ft0, ft1, %[c1] \n"
          "fmadd.s %[c2], ft0, ft1, %[c2] \n"
          "fmadd.s %[c3], ft0, ft1, %[c3] \n"
          "fmadd.s %[c4], ft0, ft1, %[c4] \n"
          "fmadd.s %[c5], ft0, ft1, %[c5] \n"
          "fmadd.s %[c6], ft0, ft1, %[c6] \n"
          "fmadd.s %[c7], ft0, ft1, %[c7] \n"
          : [c0] "+f"(c[0]), [c1] "+f"(c[1]), [c2] "+f"(c[2]), [c3] "+f"(c[3]),
            [c4] "+f"(c[4]), [c5] "+f"(c[5]), [c6] "+f"(c[6]), [c7] "+f"(c[7])
          : [n_frep] "r"(K - 1)
          : "ft0", "ft1", "ft2");

      // Store results back
      Y[C_offset + m * ldC + n + 0] = c[0];
      Y[C_offset + m * ldC + n + 1] = c[1];
      Y[C_offset + m * ldC + n + 2] = c[2];
      Y[C_offset + m * ldC + n + 3] = c[3];
      Y[C_offset + m * ldC + n + 4] = c[4];
      Y[C_offset + m * ldC + n + 5] = c[5];
      Y[C_offset + m * ldC + n + 6] = c[6];
      Y[C_offset + m * ldC + n + 7] = c[7];
      n += unroll;
    }

    // Clean up of leftover columns
    snrt_ssr_disable();
    for (; n < N; n++) {
      float32_t c;
      if (BETA) {
        c = C[C_offset + m * ldC + n];
      } else {
        c = 0.0;
      }
      for (uint32_t k = 0; k < K; k++) {
        c += A[A_offset + k + m * ldA] * B[k + n * ldB];
      }
      Y[C_offset + m * ldC + n] = c;
    }
    snrt_ssr_enable();
  }
  snrt_ssr_disable();
}

void gemm_fp32_opt(uint32_t M, uint32_t N, uint32_t K, float32_t *A,
                   uint32_t ldA, float32_t *B, uint32_t ldB, float32_t *C,
                   uint32_t ldC, float32_t *Y, uint32_t BETA,
                   uint32_t setup_SSR) {
  uint32_t compute_id = snrt_global_compute_core_idx();
  uint32_t A_offset = K * compute_id;
  uint32_t C_offset = N * compute_id;

  // Unrolling factor of most inner loop.
  // Should be at least as high as the FMA delay
  // for maximum utilization
  const uint32_t unroll = 8;

  // SSR strides and bounds only have to be configured
  // once in the beginning
  if (setup_SSR) {
    // First matrix is not stored in transposed format
    const uint32_t ssr0_b[4] = {unroll, K, N / unroll, M};
    const uint32_t ssr0_i[4] = {0, sizeof(float32_t), 0,
                                sizeof(float32_t) * ldA};

    // Second matrix is not stored in transposed format
    const uint32_t ssr1_b[4] = {unroll, K, N / unroll, M};
    const uint32_t ssr1_i[4] = {sizeof(float32_t), sizeof(float32_t) * ldB,
                                sizeof(float32_t) * unroll, 0};

    snrt_ssr_loop_3d(SNRT_SSR_DM0, ssr0_b[1], ssr0_b[2], ssr0_b[3], ssr0_i[1],
                     ssr0_i[2], ssr0_i[3]);

    snrt_ssr_repeat(SNRT_SSR_DM0, unroll);
    snrt_ssr_loop_4d(SNRT_SSR_DM1, ssr1_b[0], ssr1_b[1], ssr1_b[2], ssr1_b[3],
                     ssr1_i[0], ssr1_i[1], ssr1_i[2], ssr1_i[3]);
  }

  // SSR start address need to be configured each time

  snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, &A[A_offset]);
  snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, B);
  snrt_ssr_enable();

  // check dimensions and values of a and b

  // Kernel progresses by 1 values each step
  // const uint32_t n_frep = K - 1;
  for (uint32_t m = 0; m < M; m++) {
    uint32_t n = 0;
    for (uint32_t n0 = 0; n0 < N / unroll; n0++) {
      float c[unroll];

      // Load intermediate result
      if (BETA) {
        c[0] = C[C_offset + m * ldC + n + 0];
        c[1] = C[C_offset + m * ldC + n + 1];
        c[2] = C[C_offset + m * ldC + n + 2];
        c[3] = C[C_offset + m * ldC + n + 3];
        c[4] = C[C_offset + m * ldC + n + 4];
        c[5] = C[C_offset + m * ldC + n + 5];
        c[6] = C[C_offset + m * ldC + n + 6];
        c[7] = C[C_offset + m * ldC + n + 7];
      } else {
        c[0] = 0.0;
        c[1] = 0.0;
        c[2] = 0.0;
        c[3] = 0.0;
        c[4] = 0.0;
        c[5] = 0.0;
        c[6] = 0.0;
        c[7] = 0.0;
      }

      asm volatile(
          "frep.o %[n_frep], 8, 0, 0 \n"
          "fmadd.s %[c0], ft0, ft1, %[c0] \n"
          "fmadd.s %[c1], ft0, ft1, %[c1] \n"
          "fmadd.s %[c2], ft0, ft1, %[c2] \n"
          "fmadd.s %[c3], ft0, ft1, %[c3] \n"
          "fmadd.s %[c4], ft0, ft1, %[c4] \n"
          "fmadd.s %[c5], ft0, ft1, %[c5] \n"
          "fmadd.s %[c6], ft0, ft1, %[c6] \n"
          "fmadd.s %[c7], ft0, ft1, %[c7] \n"
          : [c0] "+f"(c[0]), [c1] "+f"(c[1]), [c2] "+f"(c[2]), [c3] "+f"(c[3]),
            [c4] "+f"(c[4]), [c5] "+f"(c[5]), [c6] "+f"(c[6]), [c7] "+f"(c[7])
          : [n_frep] "r"(K - 1)
          : "ft0", "ft1", "ft2");

      // Store results back
      Y[C_offset + m * ldC + n + 0] = c[0];
      Y[C_offset + m * ldC + n + 1] = c[1];
      Y[C_offset + m * ldC + n + 2] = c[2];
      Y[C_offset + m * ldC + n + 3] = c[3];
      Y[C_offset + m * ldC + n + 4] = c[4];
      Y[C_offset + m * ldC + n + 5] = c[5];
      Y[C_offset + m * ldC + n + 6] = c[6];
      Y[C_offset + m * ldC + n + 7] = c[7];
      n += unroll;
    }

    // Clean up of leftover columns
    snrt_ssr_disable();
    for (; n < N; n++) {
      float32_t c;
      if (BETA) {
        c = C[C_offset + m * ldC + n];
      } else {
        c = 0.0;
      }
      for (uint32_t k = 0; k < K; k++) {
        c += A[A_offset + k + m * ldA] * B[k * ldB + n];
      }
      Y[C_offset + m * ldC + n] = c;
    }
    snrt_ssr_enable();
  }
  snrt_ssr_disable();
}
