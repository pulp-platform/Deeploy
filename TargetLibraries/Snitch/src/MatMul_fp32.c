/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

/*
 * One UNROLL=8-column o-tile of the SSR+FREP matmul inner kernel.
 *
 * Zeroes 4 v2f32 accumulators, runs the FREP-replayed 4x vfmac.r.s block over
 * the N reduction dimension (consuming the live SSR ft0/ft1 streams), and
 * writes the 8 results to cptr. Both matmul_fp32_ssr_frep variants (M-parallel
 * and O-parallel) share this exact sequence; only the surrounding stream setup
 * and loop bounds differ. n_frep must be N-1 (FREP runs the body n_frep+1
 * times).
 */
static inline void _matmul_fp32_ssr_frep_otile(float32_t *cptr,
                                               uint32_t n_frep) {
  const register float zero = 0.0f;
  v2f32 c0, c1, c2, c3;
  asm volatile("vfcpka.s.s %[c0], %[z], %[z] \n"
               "vfcpka.s.s %[c1], %[z], %[z] \n"
               "vfcpka.s.s %[c2], %[z], %[z] \n"
               "vfcpka.s.s %[c3], %[z], %[z] \n"
               "frep.o %[nf], 4, 0, 0 \n"
               "vfmac.r.s %[c0], ft1, ft0 \n"
               "vfmac.r.s %[c1], ft1, ft0 \n"
               "vfmac.r.s %[c2], ft1, ft0 \n"
               "vfmac.r.s %[c3], ft1, ft0 \n"
               : [c0] "=&f"(c0), [c1] "=&f"(c1), [c2] "=&f"(c2), [c3] "=&f"(c3)
               : [z] "f"(zero), [nf] "r"(n_frep)
               : "ft0", "ft1", "ft2");
  ((v2f32 *)cptr)[0] = c0;
  ((v2f32 *)cptr)[1] = c1;
  ((v2f32 *)cptr)[2] = c2;
  ((v2f32 *)cptr)[3] = c3;
}

/*
 * Multi-core FP32 matrix multiplication (scalar, no SSR)
 *
 * Computes: Y = A * B
 * A is M x N, B is N x O, Y is M x O
 * All matrices in row-major layout.
 *
 * Splits M rows across compute cores internally.
 * Uses a distinct function name to avoid being shadowed by
 * the Generic single-core MatMul_fp32_fp32_fp32 (link order).
 */
void matmul_fp32_opt(const float32_t *__restrict__ pSrcA,
                     const float32_t *__restrict__ pSrcB,
                     float32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                     uint32_t O) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t rows_per_core = M / numThreads;
  uint32_t remainder = M % numThreads;

  uint32_t start_row, num_rows;
  if (core_id < remainder) {
    num_rows = rows_per_core + 1;
    start_row = core_id * num_rows;
  } else {
    num_rows = rows_per_core;
    start_row = core_id * rows_per_core + remainder;
  }

  const uint32_t unroll = 8;
  uint32_t O_block = O - (O % unroll);

  for (uint32_t i = start_row; i < start_row + num_rows; i++) {
    uint32_t j;
    for (j = 0; j < O_block; j += unroll) {
      float32_t c0 = 0.0f;
      float32_t c1 = 0.0f;
      float32_t c2 = 0.0f;
      float32_t c3 = 0.0f;
      float32_t c4 = 0.0f;
      float32_t c5 = 0.0f;
      float32_t c6 = 0.0f;
      float32_t c7 = 0.0f;

      for (uint32_t k = 0; k < N; k++) {
        float32_t a = pSrcA[i * N + k];
        c0 += a * pSrcB[k * O + j + 0];
        c1 += a * pSrcB[k * O + j + 1];
        c2 += a * pSrcB[k * O + j + 2];
        c3 += a * pSrcB[k * O + j + 3];
        c4 += a * pSrcB[k * O + j + 4];
        c5 += a * pSrcB[k * O + j + 5];
        c6 += a * pSrcB[k * O + j + 6];
        c7 += a * pSrcB[k * O + j + 7];
      }

      pDstY[i * O + j + 0] = c0;
      pDstY[i * O + j + 1] = c1;
      pDstY[i * O + j + 2] = c2;
      pDstY[i * O + j + 3] = c3;
      pDstY[i * O + j + 4] = c4;
      pDstY[i * O + j + 5] = c5;
      pDstY[i * O + j + 6] = c6;
      pDstY[i * O + j + 7] = c7;
    }

    // Cleanup for remaining columns
    for (; j < O; j++) {
      float32_t sum = 0.0f;
      for (uint32_t k = 0; k < N; k++) {
        sum += pSrcA[i * N + k] * pSrcB[k * O + j];
      }
      pDstY[i * O + j] = sum;
    }
  }
}

/*
 * Multi-core FP32 matrix multiplication with SSR + FREP (v2f32 SIMD)
 *
 * Computes: Y = A * B
 * A is M x N, B is N x O, Y is M x O, all row-major, contiguous.
 *
 * Splits M rows across compute cores internally. Each core configures its own
 * SSR streams and processes its contiguous block of rows.
 *
 * O-dimension pair-packing: B[k, 2u] and B[k, 2u+1] are consecutive in
 * row-major B, so a v2f32 load is correct. Each o-tile of UNROLL=8 columns is
 * handled as UNROLL/2 = 4 v2f32 accumulators.
 *   SSR DM0 (ft0): streams A scalars, each held UNROLL/2 times (repeat).
 *   SSR DM1 (ft1): streams B as v2f32 pairs.
 *   FREP: replays the 4x vfmac.r.s block over the N (reduction) dimension;
 *         vfmac.r.s broadcasts the A scalar across both lanes of each v2f32.
 *
 * NOTE: Snitch SSR can only stream from cluster TCDM/L1, so operands must be
 * staged into L1 (tiled flow) before calling this kernel.
 *
 * Leftover columns (O % UNROLL) fall back to a scalar reduction.
 */
void matmul_fp32_ssr_frep(const float32_t *__restrict__ pSrcA,
                          const float32_t *__restrict__ pSrcB,
                          float32_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                          uint32_t O) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  uint32_t rows_per_core = M / numThreads;
  uint32_t remainder = M % numThreads;

  uint32_t start_row, num_rows;
  if (core_id < remainder) {
    num_rows = rows_per_core + 1;
    start_row = core_id * num_rows;
  } else {
    num_rows = rows_per_core;
    start_row = core_id * rows_per_core + remainder;
  }

  if (num_rows == 0)
    return;

  const uint32_t UNROLL = 8;
  const uint32_t UNROLL_PAIRS = UNROLL / 2; // 4 v2f32 accumulators per o-tile
  uint32_t O_tiles = O / UNROLL;

  const uint32_t ldA = N, ldB = O, ldC = O;
  const float32_t *A = &pSrcA[start_row * ldA];
  float32_t *C = &pDstY[start_row * ldC];

  if (O_tiles > 0) {
    // DM0 = A: bounds (N, O_tiles, num_rows), each element repeated
    // UNROLL_PAIRS
    snrt_ssr_loop_3d(SNRT_SSR_DM0, N, O_tiles, num_rows, sizeof(float32_t), 0,
                     sizeof(float32_t) * ldA);
    snrt_ssr_repeat(SNRT_SSR_DM0, UNROLL_PAIRS);

    // DM1 = B: v2f32 pairs (8-byte elements)
    snrt_ssr_loop_4d(SNRT_SSR_DM1, UNROLL_PAIRS, N, O_tiles, num_rows,
                     sizeof(float32_t) * 2, sizeof(float32_t) * ldB,
                     sizeof(float32_t) * UNROLL, 0);

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void *)A);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void *)pSrcB);
    snrt_ssr_enable();

    const uint32_t n_frep = N - 1;

    for (uint32_t m = 0; m < num_rows; m++) {
      for (uint32_t o0 = 0; o0 < O_tiles; o0++) {
        _matmul_fp32_ssr_frep_otile(&C[m * ldC + o0 * UNROLL], n_frep);
      }
    }
    snrt_ssr_disable();
  }

  // Scalar fallback for leftover columns (O % UNROLL)
  uint32_t o_done = O_tiles * UNROLL;
  if (o_done < O) {
    for (uint32_t m = 0; m < num_rows; m++) {
      uint32_t row = start_row + m;
      for (uint32_t o = o_done; o < O; o++) {
        float32_t acc = 0.0f;
        for (uint32_t k = 0; k < N; k++) {
          acc += pSrcA[row * N + k] * pSrcB[k * O + o];
        }
        pDstY[row * O + o] = acc;
      }
    }
  }
}

/*
 * Multi-core FP32 matrix multiplication with SSR + FREP, parallelized over the
 * O (output column) dimension instead of M (rows).
 *
 * Motivation: in autoregressive decode every MatMul has M=1, so the M-parallel
 * matmul_fp32_ssr_frep leaves 7/8 cores idle. Splitting the O tiles across
 * cores keeps all cores busy when M=1 (and is equivalent work for larger M).
 *
 * Each core owns a contiguous range of UNROLL=8-column O-tiles and computes all
 * M rows for those columns. Same v2f32 SSR + FREP inner kernel (normal stores,
 * no DM2 write stream). O % UNROLL leftover columns are handled by core 0.
 *
 * NOTE: Snitch SSR can only stream from cluster TCDM/L1 (tiled flow).
 */
void matmul_fp32_ssr_frep_oparallel(const float32_t *__restrict__ pSrcA,
                                    const float32_t *__restrict__ pSrcB,
                                    float32_t *__restrict__ pDstY, uint32_t M,
                                    uint32_t N, uint32_t O) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  const uint32_t UNROLL = 8;
  const uint32_t UNROLL_PAIRS = UNROLL / 2;
  uint32_t O_tiles = O / UNROLL;

  // Split O-tiles across cores (so M=1 still uses every core).
  uint32_t tiles_per_core = O_tiles / numThreads;
  uint32_t rem_tiles = O_tiles % numThreads;
  uint32_t ot_start, num_otiles;
  if (core_id < rem_tiles) {
    num_otiles = tiles_per_core + 1;
    ot_start = core_id * num_otiles;
  } else {
    num_otiles = tiles_per_core;
    ot_start = core_id * tiles_per_core + rem_tiles;
  }

  const uint32_t ldA = N, ldB = O, ldC = O;

  if (num_otiles > 0) {
    uint32_t o_base = ot_start * UNROLL; // first column this core owns

    // DM0 = A: (N, num_otiles, M); A reused across o-tiles (stride 0).
    snrt_ssr_loop_3d(SNRT_SSR_DM0, N, num_otiles, M, sizeof(float32_t), 0,
                     sizeof(float32_t) * ldA);
    snrt_ssr_repeat(SNRT_SSR_DM0, UNROLL_PAIRS);

    // DM1 = B: v2f32 pairs over this core's O-chunk.
    snrt_ssr_loop_4d(SNRT_SSR_DM1, UNROLL_PAIRS, N, num_otiles, M,
                     sizeof(float32_t) * 2, sizeof(float32_t) * ldB,
                     sizeof(float32_t) * UNROLL, 0);

    snrt_ssr_read(SNRT_SSR_DM0, SNRT_SSR_4D, (void *)pSrcA);
    snrt_ssr_read(SNRT_SSR_DM1, SNRT_SSR_4D, (void *)&pSrcB[o_base]);
    snrt_ssr_enable();

    const uint32_t n_frep = N - 1;

    for (uint32_t m = 0; m < M; m++) {
      for (uint32_t o0 = 0; o0 < num_otiles; o0++) {
        _matmul_fp32_ssr_frep_otile(&pDstY[m * ldC + o_base + o0 * UNROLL],
                                    n_frep);
      }
    }
    snrt_ssr_disable();
  }

  // Leftover O % UNROLL columns: core 0 handles them (scalar) for all rows.
  uint32_t o_done = O_tiles * UNROLL;
  if (o_done < O && core_id == 0) {
    for (uint32_t m = 0; m < M; m++) {
      for (uint32_t o = o_done; o < O; o++) {
        float32_t acc = 0.0f;
        for (uint32_t k = 0; k < N; k++) {
          acc += pSrcA[m * N + k] * pSrcB[k * O + o];
        }
        pDstY[m * O + o] = acc;
      }
    }
  }
}
