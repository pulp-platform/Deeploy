/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"
// #include "perf_utils.h"

void PULP_Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                                   const float32_t *__restrict__ pSrcB,
                                   const float32_t *__restrict__ pDstC,
                                   float32_t *__restrict__ pDstY, uint32_t M,
                                   uint32_t N, uint32_t O, uint32_t transA,
                                   uint32_t transB) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  //RW: Performance monitoring is currently disabled 
  // perf_stats_t perf_start, perf_end, perf_total;

  // // Initialize and start performance counters (only core 0)
  // if (core_id == 0) {
  //   perf_bench_init();
  //   perf_bench_start();
  //   perf_bench_read(&perf_start);
  // }

  uint32_t M_chunk = (M >> log2Core) + ((M & (NUM_CORES - 1)) != 0);
  uint32_t M_start = MIN(core_id * M_chunk, M);
  uint32_t M_end = MIN(M_start + M_chunk, M);
  uint32_t M_size = M_end - M_start;

  if (M_size == 0) {
    return;
  }

  const uint32_t has_bias = (pDstC != NULL);
  const uint32_t N_unroll = N - (N % 6);
  const uint32_t O_unroll = O - (O % 6);

  if (!transA && !transB) {

    for (uint32_t i = M_start; i < M_end; ++i) {
      const float32_t *__restrict__ a_row = &pSrcA[i * N];
      float32_t *__restrict__ y_row = &pDstY[i * O];
      const float32_t *__restrict__ c_row = has_bias ? &pDstC[i * O] : NULL;

      uint32_t j = 0;

      for (; j < O_unroll; j += 6) {
        float32_t sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f,
                  sum4 = 0.0f, sum5 = 0.0f;

        uint32_t k = 0;

        for (; k < N; ++k) {
          const float32_t a_val = a_row[k];
          sum0 += a_val * pSrcB[k * O + j];
          sum1 += a_val * pSrcB[k * O + j + 1];
          sum2 += a_val * pSrcB[k * O + j + 2];
          sum3 += a_val * pSrcB[k * O + j + 3];
          sum4 += a_val * pSrcB[k * O + j + 4];
          sum5 += a_val * pSrcB[k * O + j + 5];
        }

        if (has_bias) {
          y_row[j] = sum0 + c_row[j];
          y_row[j + 1] = sum1 + c_row[j + 1];
          y_row[j + 2] = sum2 + c_row[j + 2];
          y_row[j + 3] = sum3 + c_row[j + 3];
          y_row[j + 4] = sum4 + c_row[j + 4];
          y_row[j + 5] = sum5 + c_row[j + 5];
        } else {
          y_row[j] = sum0;
          y_row[j + 1] = sum1;
          y_row[j + 2] = sum2;
          y_row[j + 3] = sum3;
          y_row[j + 4] = sum4;
          y_row[j + 5] = sum5;
        }
      }

      for (; j < O; ++j) {
        float32_t sum = 0.0f;
        for (uint32_t k = 0; k < N; ++k) {
          sum += a_row[k] * pSrcB[k * O + j];
        }

        y_row[j] = has_bias ? sum + c_row[j] : sum;
      }
    }
  } else if (transA && !transB) {

    for (uint32_t i = M_start; i < M_end; ++i) {
      float32_t *__restrict__ y_row = &pDstY[i * O];
      const float32_t *__restrict__ c_row = has_bias ? &pDstC[i * O] : NULL;

      uint32_t j = 0;
      for (; j < O_unroll; j += 6) {
        float32_t sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f,
                  sum4 = 0.0f, sum5 = 0.0f;

        uint32_t k = 0;
        for (; k < N_unroll; k += 6) {
          const float32_t a0 = pSrcA[k * M + i];
          const float32_t a1 = pSrcA[(k + 1) * M + i];
          const float32_t a2 = pSrcA[(k + 2) * M + i];
          const float32_t a3 = pSrcA[(k + 3) * M + i];
          const float32_t a4 = pSrcA[(k + 4) * M + i];
          const float32_t a5 = pSrcA[(k + 5) * M + i];

          sum0 += a0 * pSrcB[k * O + j] + a1 * pSrcB[(k + 1) * O + j] +
                  a2 * pSrcB[(k + 2) * O + j] + a3 * pSrcB[(k + 3) * O + j] +
                  a4 * pSrcB[(k + 4) * O + j] + a5 * pSrcB[(k + 5) * O + j];
          sum1 += a0 * pSrcB[k * O + j + 1] + a1 * pSrcB[(k + 1) * O + j + 1] +
                  a2 * pSrcB[(k + 2) * O + j + 1] +
                  a3 * pSrcB[(k + 3) * O + j + 1] +
                  a4 * pSrcB[(k + 4) * O + j + 1] +
                  a5 * pSrcB[(k + 5) * O + j + 1];
          sum2 += a0 * pSrcB[k * O + j + 2] + a1 * pSrcB[(k + 1) * O + j + 2] +
                  a2 * pSrcB[(k + 2) * O + j + 2] +
                  a3 * pSrcB[(k + 3) * O + j + 2] +
                  a4 * pSrcB[(k + 4) * O + j + 2] +
                  a5 * pSrcB[(k + 5) * O + j + 2];
          sum3 += a0 * pSrcB[k * O + j + 3] + a1 * pSrcB[(k + 1) * O + j + 3] +
                  a2 * pSrcB[(k + 2) * O + j + 3] +
                  a3 * pSrcB[(k + 3) * O + j + 3] +
                  a4 * pSrcB[(k + 4) * O + j + 3] +
                  a5 * pSrcB[(k + 5) * O + j + 3];
          sum4 += a0 * pSrcB[k * O + j + 4] + a1 * pSrcB[(k + 1) * O + j + 4] +
                  a2 * pSrcB[(k + 2) * O + j + 4] +
                  a3 * pSrcB[(k + 3) * O + j + 4] +
                  a4 * pSrcB[(k + 4) * O + j + 4] +
                  a5 * pSrcB[(k + 5) * O + j + 4];
          sum5 += a0 * pSrcB[k * O + j + 5] + a1 * pSrcB[(k + 1) * O + j + 5] +
                  a2 * pSrcB[(k + 2) * O + j + 5] +
                  a3 * pSrcB[(k + 3) * O + j + 5] +
                  a4 * pSrcB[(k + 4) * O + j + 5] +
                  a5 * pSrcB[(k + 5) * O + j + 5];
        }

        for (; k < N; ++k) {
          const float32_t a_val = pSrcA[k * M + i];
          sum0 += a_val * pSrcB[k * O + j];
          sum1 += a_val * pSrcB[k * O + j + 1];
          sum2 += a_val * pSrcB[k * O + j + 2];
          sum3 += a_val * pSrcB[k * O + j + 3];
          sum4 += a_val * pSrcB[k * O + j + 4];
          sum5 += a_val * pSrcB[k * O + j + 5];
        }

        if (has_bias) {
          y_row[j] = sum0 + c_row[j];
          y_row[j + 1] = sum1 + c_row[j + 1];
          y_row[j + 2] = sum2 + c_row[j + 2];
          y_row[j + 3] = sum3 + c_row[j + 3];
          y_row[j + 4] = sum4 + c_row[j + 4];
          y_row[j + 5] = sum5 + c_row[j + 5];
        } else {
          y_row[j] = sum0;
          y_row[j + 1] = sum1;
          y_row[j + 2] = sum2;
          y_row[j + 3] = sum3;
          y_row[j + 4] = sum4;
          y_row[j + 5] = sum5;
        }
      }

      for (; j < O; ++j) {
        float32_t sum = 0.0f;
        uint32_t k = 0;
        for (; k < N_unroll; k += 6) {
          sum += pSrcA[k * M + i] * pSrcB[k * O + j] +
                 pSrcA[(k + 1) * M + i] * pSrcB[(k + 1) * O + j] +
                 pSrcA[(k + 2) * M + i] * pSrcB[(k + 2) * O + j] +
                 pSrcA[(k + 3) * M + i] * pSrcB[(k + 3) * O + j] +
                 pSrcA[(k + 4) * M + i] * pSrcB[(k + 4) * O + j] +
                 pSrcA[(k + 5) * M + i] * pSrcB[(k + 5) * O + j];
        }
        for (; k < N; ++k) {
          sum += pSrcA[k * M + i] * pSrcB[k * O + j];
        }

        y_row[j] = has_bias ? sum + c_row[j] : sum;
      }
    }
  } else if (!transA && transB) {

    for (uint32_t i = M_start; i < M_end; ++i) {
      const float32_t *__restrict__ a_row = &pSrcA[i * N];
      float32_t *__restrict__ y_row = &pDstY[i * O];
      const float32_t *__restrict__ c_row = has_bias ? &pDstC[i * O] : NULL;

      uint32_t j = 0;
      for (; j < O_unroll; j += 6) {
        const float32_t *__restrict__ b_row0 = &pSrcB[j * N];
        const float32_t *__restrict__ b_row1 = &pSrcB[(j + 1) * N];
        const float32_t *__restrict__ b_row2 = &pSrcB[(j + 2) * N];
        const float32_t *__restrict__ b_row3 = &pSrcB[(j + 3) * N];
        const float32_t *__restrict__ b_row4 = &pSrcB[(j + 4) * N];
        const float32_t *__restrict__ b_row5 = &pSrcB[(j + 5) * N];

        float32_t sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f,
                  sum4 = 0.0f, sum5 = 0.0f;

        uint32_t k = 0;
        for (; k < N_unroll; k += 6) {
          const float32_t a0 = a_row[k];
          const float32_t a1 = a_row[k + 1];
          const float32_t a2 = a_row[k + 2];
          const float32_t a3 = a_row[k + 3];
          const float32_t a4 = a_row[k + 4];
          const float32_t a5 = a_row[k + 5];

          sum0 += a0 * b_row0[k] + a1 * b_row0[k + 1] + a2 * b_row0[k + 2] +
                  a3 * b_row0[k + 3] + a4 * b_row0[k + 4] + a5 * b_row0[k + 5];
          sum1 += a0 * b_row1[k] + a1 * b_row1[k + 1] + a2 * b_row1[k + 2] +
                  a3 * b_row1[k + 3] + a4 * b_row1[k + 4] + a5 * b_row1[k + 5];
          sum2 += a0 * b_row2[k] + a1 * b_row2[k + 1] + a2 * b_row2[k + 2] +
                  a3 * b_row2[k + 3] + a4 * b_row2[k + 4] + a5 * b_row2[k + 5];
          sum3 += a0 * b_row3[k] + a1 * b_row3[k + 1] + a2 * b_row3[k + 2] +
                  a3 * b_row3[k + 3] + a4 * b_row3[k + 4] + a5 * b_row3[k + 5];
          sum4 += a0 * b_row4[k] + a1 * b_row4[k + 1] + a2 * b_row4[k + 2] +
                  a3 * b_row4[k + 3] + a4 * b_row4[k + 4] + a5 * b_row4[k + 5];
          sum5 += a0 * b_row5[k] + a1 * b_row5[k + 1] + a2 * b_row5[k + 2] +
                  a3 * b_row5[k + 3] + a4 * b_row5[k + 4] + a5 * b_row5[k + 5];
        }

        for (; k < N; ++k) {
          const float32_t a_val = a_row[k];
          sum0 += a_val * b_row0[k];
          sum1 += a_val * b_row1[k];
          sum2 += a_val * b_row2[k];
          sum3 += a_val * b_row3[k];
          sum4 += a_val * b_row4[k];
          sum5 += a_val * b_row5[k];
        }

        if (has_bias) {
          y_row[j] = sum0 + c_row[j];
          y_row[j + 1] = sum1 + c_row[j + 1];
          y_row[j + 2] = sum2 + c_row[j + 2];
          y_row[j + 3] = sum3 + c_row[j + 3];
          y_row[j + 4] = sum4 + c_row[j + 4];
          y_row[j + 5] = sum5 + c_row[j + 5];
        } else {
          y_row[j] = sum0;
          y_row[j + 1] = sum1;
          y_row[j + 2] = sum2;
          y_row[j + 3] = sum3;
          y_row[j + 4] = sum4;
          y_row[j + 5] = sum5;
        }
      }

      for (; j < O; ++j) {
        const float32_t *__restrict__ b_row = &pSrcB[j * N];
        float32_t sum = 0.0f;

        uint32_t k = 0;
        for (; k < N_unroll; k += 6) {
          sum += a_row[k] * b_row[k] + a_row[k + 1] * b_row[k + 1] +
                 a_row[k + 2] * b_row[k + 2] + a_row[k + 3] * b_row[k + 3] +
                 a_row[k + 4] * b_row[k + 4] + a_row[k + 5] * b_row[k + 5];
        }
        for (; k < N; ++k) {
          sum += a_row[k] * b_row[k];
        }

        y_row[j] = has_bias ? sum + c_row[j] : sum;
      }
    }
  } else {

    for (uint32_t i = M_start; i < M_end; ++i) {
      float32_t *__restrict__ y_row = &pDstY[i * O];
      const float32_t *__restrict__ c_row = has_bias ? &pDstC[i * O] : NULL;

      uint32_t j = 0;
      for (; j < O_unroll; j += 6) {
        const float32_t *__restrict__ b_row0 = &pSrcB[j * N];
        const float32_t *__restrict__ b_row1 = &pSrcB[(j + 1) * N];
        const float32_t *__restrict__ b_row2 = &pSrcB[(j + 2) * N];
        const float32_t *__restrict__ b_row3 = &pSrcB[(j + 3) * N];
        const float32_t *__restrict__ b_row4 = &pSrcB[(j + 4) * N];
        const float32_t *__restrict__ b_row5 = &pSrcB[(j + 5) * N];

        float32_t sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f,
                  sum4 = 0.0f, sum5 = 0.0f;

        uint32_t k = 0;
        for (; k < N_unroll; k += 6) {
          const float32_t a0 = pSrcA[k * M + i];
          const float32_t a1 = pSrcA[(k + 1) * M + i];
          const float32_t a2 = pSrcA[(k + 2) * M + i];
          const float32_t a3 = pSrcA[(k + 3) * M + i];
          const float32_t a4 = pSrcA[(k + 4) * M + i];
          const float32_t a5 = pSrcA[(k + 5) * M + i];

          sum0 += a0 * b_row0[k] + a1 * b_row0[k + 1] + a2 * b_row0[k + 2] +
                  a3 * b_row0[k + 3] + a4 * b_row0[k + 4] + a5 * b_row0[k + 5];
          sum1 += a0 * b_row1[k] + a1 * b_row1[k + 1] + a2 * b_row1[k + 2] +
                  a3 * b_row1[k + 3] + a4 * b_row1[k + 4] + a5 * b_row1[k + 5];
          sum2 += a0 * b_row2[k] + a1 * b_row2[k + 1] + a2 * b_row2[k + 2] +
                  a3 * b_row2[k + 3] + a4 * b_row2[k + 4] + a5 * b_row2[k + 5];
          sum3 += a0 * b_row3[k] + a1 * b_row3[k + 1] + a2 * b_row3[k + 2] +
                  a3 * b_row3[k + 3] + a4 * b_row3[k + 4] + a5 * b_row3[k + 5];
          sum4 += a0 * b_row4[k] + a1 * b_row4[k + 1] + a2 * b_row4[k + 2] +
                  a3 * b_row4[k + 3] + a4 * b_row4[k + 4] + a5 * b_row4[k + 5];
          sum5 += a0 * b_row5[k] + a1 * b_row5[k + 1] + a2 * b_row5[k + 2] +
                  a3 * b_row5[k + 3] + a4 * b_row5[k + 4] + a5 * b_row5[k + 5];
        }

        for (; k < N; ++k) {
          const float32_t a_val = pSrcA[k * M + i];
          sum0 += a_val * b_row0[k];
          sum1 += a_val * b_row1[k];
          sum2 += a_val * b_row2[k];
          sum3 += a_val * b_row3[k];
          sum4 += a_val * b_row4[k];
          sum5 += a_val * b_row5[k];
        }

        if (has_bias) {
          y_row[j] = sum0 + c_row[j];
          y_row[j + 1] = sum1 + c_row[j + 1];
          y_row[j + 2] = sum2 + c_row[j + 2];
          y_row[j + 3] = sum3 + c_row[j + 3];
          y_row[j + 4] = sum4 + c_row[j + 4];
          y_row[j + 5] = sum5 + c_row[j + 5];
        } else {
          y_row[j] = sum0;
          y_row[j + 1] = sum1;
          y_row[j + 2] = sum2;
          y_row[j + 3] = sum3;
          y_row[j + 4] = sum4;
          y_row[j + 5] = sum5;
        }
      }

      for (; j < O; ++j) {
        const float32_t *__restrict__ b_row = &pSrcB[j * N];
        float32_t sum = 0.0f;

        uint32_t k = 0;
        for (; k < N_unroll; k += 6) {
          sum += pSrcA[k * M + i] * b_row[k] +
                 pSrcA[(k + 1) * M + i] * b_row[k + 1] +
                 pSrcA[(k + 2) * M + i] * b_row[k + 2] +
                 pSrcA[(k + 3) * M + i] * b_row[k + 3] +
                 pSrcA[(k + 4) * M + i] * b_row[k + 4] +
                 pSrcA[(k + 5) * M + i] * b_row[k + 5];
        }
        for (; k < N; ++k) {
          sum += pSrcA[k * M + i] * b_row[k];
        }

        y_row[j] = has_bias ? sum + c_row[j] : sum;
      }
    }
  }

  // RW: Stop performance counters and print results (only core 0)
  // if (core_id == 0) {
  //   perf_bench_stop();
  //   perf_bench_read(&perf_end);
  //   perf_bench_diff(&perf_total, &perf_end, &perf_start);

  //   char label[100];
  //   snprintf(label, sizeof(label), "GEMM M=%u N=%u O=%u transA=%u transB=%u",
  //            M, N, O, transA, transB);
  //   perf_bench_print(label, &perf_total);
  // }
}