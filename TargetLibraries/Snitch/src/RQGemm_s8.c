/* =====================================================================
 * Title:        RQGemm_s8.c
 * Description:
 *
 * Date:         30.05.2024
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeploySnitchMath.h"
#include "RQGemm.h"

// Assumptions:
//  - per-row requantization
//  - single batch
void RQGemm_s8_row_parallel(int8_t const *__restrict__ pSrcA,
                            int8_t const *__restrict__ pSrcB,
                            int32_t const *__restrict__ pSrcC,
                            int8_t *__restrict__ pDstY, uint32_t M, uint32_t N,
                            uint32_t O, int32_t alpha, int32_t beta,
                            int32_t *mul, int32_t *add, int32_t log2D) {
  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core a row tile
  uint32_t const MQuotient = M / numThreads;
  uint32_t const MRemainder = M % numThreads;
  uint32_t const MSize = MQuotient + (core_id < MRemainder ? 1 : 0);
  uint32_t const MStart =
      core_id * MQuotient + (core_id < MRemainder ? core_id : MRemainder);
  uint32_t const MEnd = MStart + MSize;

  if (core_id < numThreads) {
    for (uint32_t m = MStart; m < MEnd; m++) {
      for (uint32_t o = 0; o < O; o++) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)pSrcA[m * N + n] * pSrcB[n * O + o];
        }
        sum = alpha * sum + beta * pSrcC[m * O + o];

        // Requantize value
        sum = (sum * mul[m] + add[m]) >> log2D;
        pDstY[m * O + o] = (int8_t)CLAMP(sum, -128, 127);
      }
    }
  }
}

// Assumptions:
//  - per-row requantization
//  - transposed input B
//  - single batch
void RQGemm_s8_row_parallel_unrolled(int8_t const *__restrict__ pSrcA,
                                     int8_t const *__restrict__ pSrcB,
                                     int32_t const *__restrict__ pSrcC,
                                     int8_t *__restrict__ pDstY, uint32_t M,
                                     uint32_t N, uint32_t O, int32_t alpha,
                                     int32_t beta, int32_t *mul, int32_t *add,
                                     int32_t log2D) {
  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core a row tile
  uint32_t const MQuotient = M / numThreads;
  uint32_t const MRemainder = M % numThreads;
  uint32_t const MSize = MQuotient + (core_id < MRemainder ? 1 : 0);
  uint32_t const MStart =
      core_id * MQuotient + (core_id < MRemainder ? core_id : MRemainder);
  uint32_t const MEnd = MStart + MSize;

  if (core_id < numThreads) {
    for (uint32_t m = MStart; m + 1 < MEnd; m += 2) {
      for (uint32_t o = 0; o + 1 < O; o += 2) {
        int32_t sum0 = 0;
        int32_t sum1 = 0;
        int32_t sum2 = 0;
        int32_t sum3 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[n * O + (o + 0)];
          sum1 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[n * O + (o + 1)];
          sum2 += (int32_t)pSrcA[(m + 1) * N + n] * pSrcB[n * O + (o + 0)];
          sum3 += (int32_t)pSrcA[(m + 1) * N + n] * pSrcB[n * O + (o + 1)];
        }
        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (o + 0)];
        sum1 = alpha * sum1 + beta * pSrcC[(m + 0) * O + (o + 1)];
        sum2 = alpha * sum2 + beta * pSrcC[(m + 1) * O + (o + 0)];
        sum3 = alpha * sum3 + beta * pSrcC[(m + 1) * O + (o + 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        sum1 = (sum1 * mul[m + 0] + add[m + 0]) >> log2D;
        sum2 = (sum2 * mul[m + 1] + add[m + 1]) >> log2D;
        sum3 = (sum3 * mul[m + 1] + add[m + 1]) >> log2D;
        pDstY[(m + 0) * O + (o + 0)] = (int8_t)CLAMP(sum0, -128, 127);
        pDstY[(m + 0) * O + (o + 1)] = (int8_t)CLAMP(sum1, -128, 127);
        pDstY[(m + 1) * O + (o + 0)] = (int8_t)CLAMP(sum2, -128, 127);
        pDstY[(m + 1) * O + (o + 1)] = (int8_t)CLAMP(sum3, -128, 127);
      }

      if (O % 2 == 1) {
        int32_t sum0 = 0;
        int32_t sum1 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[n * O + (O - 1)];
          sum1 += (int32_t)pSrcA[(m + 1) * N + n] * pSrcB[n * O + (O - 1)];
        }

        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (O - 1)];
        sum1 = alpha * sum1 + beta * pSrcC[(m + 1) * O + (O - 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        sum1 = (sum1 * mul[m + 1] + add[m + 1]) >> log2D;
        pDstY[(m + 0) * O + (O - 1)] = (int8_t)CLAMP(sum0, -128, 127);
        pDstY[(m + 1) * O + (O - 1)] = (int8_t)CLAMP(sum1, -128, 127);
      }
    }

    if (MSize % 2 == 1) {
      uint32_t m = MEnd - 1;

      for (uint32_t o = 0; o + 1 < O; o += 2) {
        int32_t sum0 = 0;
        int32_t sum1 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[n * O + (o + 0)];
          sum1 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[n * O + (o + 1)];
        }
        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (o + 0)];
        sum1 = alpha * sum1 + beta * pSrcC[(m + 0) * O + (o + 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        sum1 = (sum1 * mul[m + 0] + add[m + 0]) >> log2D;
        pDstY[(m + 0) * O + (o + 0)] = (int8_t)CLAMP(sum0, -128, 127);
        pDstY[(m + 0) * O + (o + 1)] = (int8_t)CLAMP(sum1, -128, 127);
      }

      if (O % 2 == 1) {
        int32_t sum0 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[n * O + (O - 1)];
        }

        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (O - 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        pDstY[(m + 0) * O + (O - 1)] = (int8_t)CLAMP(sum0, -128, 127);
      }
    }
  }
}

// Assumptions:
//  - per-row requantization
//  - transposed input B
//  - single batch
void RQGemm_s8_transB_row_parallel(int8_t const *__restrict__ pSrcA,
                                   int8_t const *__restrict__ pSrcB,
                                   int32_t const *__restrict__ pSrcC,
                                   int8_t *__restrict__ pDstY, uint32_t M,
                                   uint32_t N, uint32_t O, int32_t alpha,
                                   int32_t beta, int32_t *mul, int32_t *add,
                                   int32_t log2D) {
  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core a row tile
  uint32_t const MQuotient = M / numThreads;
  uint32_t const MRemainder = M % numThreads;
  uint32_t const MSize = MQuotient + (core_id < MRemainder ? 1 : 0);
  uint32_t const MStart =
      core_id * MQuotient + (core_id < MRemainder ? core_id : MRemainder);
  uint32_t const MEnd = MStart + MSize;

  if (core_id < numThreads) {
    for (uint32_t m = MStart; m < MEnd; m++) {
      for (uint32_t o = 0; o < O; o++) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)pSrcA[m * N + n] * pSrcB[o * N + n];
        }
        sum = alpha * sum + beta * pSrcC[m * O + o];

        // Requantize value
        sum = (sum * mul[m] + add[m]) >> log2D;
        pDstY[m * O + o] = (int8_t)CLAMP(sum, -128, 127);
      }
    }
  }
}

// Assumptions:
//  - per-row requantization
//  - transposed input B
//  - single batch
void RQGemm_s8_transB_row_parallel_unrolled(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t O, int32_t alpha, int32_t beta, int32_t *mul,
    int32_t *add, int32_t log2D) {
  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core a row tile
  uint32_t const MQuotient = M / numThreads;
  uint32_t const MRemainder = M % numThreads;
  uint32_t const MSize = MQuotient + (core_id < MRemainder ? 1 : 0);
  uint32_t const MStart =
      core_id * MQuotient + (core_id < MRemainder ? core_id : MRemainder);
  uint32_t const MEnd = MStart + MSize;

  if (core_id < numThreads) {
    for (uint32_t m = MStart; m + 1 < MEnd; m += 2) {
      for (uint32_t o = 0; o + 1 < O; o += 2) {
        int32_t sum0 = 0;
        int32_t sum1 = 0;
        int32_t sum2 = 0;
        int32_t sum3 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[(o + 0) * N + n];
          sum1 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[(o + 1) * N + n];
          sum2 += (int32_t)pSrcA[(m + 1) * N + n] * pSrcB[(o + 0) * N + n];
          sum3 += (int32_t)pSrcA[(m + 1) * N + n] * pSrcB[(o + 1) * N + n];
        }
        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (o + 0)];
        sum1 = alpha * sum1 + beta * pSrcC[(m + 0) * O + (o + 1)];
        sum2 = alpha * sum2 + beta * pSrcC[(m + 1) * O + (o + 0)];
        sum3 = alpha * sum3 + beta * pSrcC[(m + 1) * O + (o + 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        sum1 = (sum1 * mul[m + 0] + add[m + 0]) >> log2D;
        sum2 = (sum2 * mul[m + 1] + add[m + 1]) >> log2D;
        sum3 = (sum3 * mul[m + 1] + add[m + 1]) >> log2D;
        pDstY[(m + 0) * O + (o + 0)] = (int8_t)CLAMP(sum0, -128, 127);
        pDstY[(m + 0) * O + (o + 1)] = (int8_t)CLAMP(sum1, -128, 127);
        pDstY[(m + 1) * O + (o + 0)] = (int8_t)CLAMP(sum2, -128, 127);
        pDstY[(m + 1) * O + (o + 1)] = (int8_t)CLAMP(sum3, -128, 127);
      }

      if (O % 2 == 1) {
        int32_t sum0 = 0;
        int32_t sum1 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[(O - 1) * N + n];
          sum1 += (int32_t)pSrcA[(m + 1) * N + n] * pSrcB[(O - 1) * N + n];
        }

        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (O - 1)];
        sum1 = alpha * sum1 + beta * pSrcC[(m + 1) * O + (O - 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        sum1 = (sum1 * mul[m + 1] + add[m + 1]) >> log2D;
        pDstY[(m + 0) * O + (O - 1)] = (int8_t)CLAMP(sum0, -128, 127);
        pDstY[(m + 1) * O + (O - 1)] = (int8_t)CLAMP(sum1, -128, 127);
      }
    }

    if (MSize % 2 == 1) {
      uint32_t m = MEnd - 1;

      for (uint32_t o = 0; o + 1 < O; o += 2) {
        int32_t sum0 = 0;
        int32_t sum1 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[(o + 0) * N + n];
          sum1 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[(o + 1) * N + n];
        }
        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (o + 0)];
        sum1 = alpha * sum1 + beta * pSrcC[(m + 0) * O + (o + 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        sum1 = (sum1 * mul[m + 0] + add[m + 0]) >> log2D;
        pDstY[(m + 0) * O + (o + 0)] = (int8_t)CLAMP(sum0, -128, 127);
        pDstY[(m + 0) * O + (o + 1)] = (int8_t)CLAMP(sum1, -128, 127);
      }

      if (O % 2 == 1) {
        int32_t sum0 = 0;
#pragma unroll 2
        for (uint32_t n = 0; n < N; ++n) {
          sum0 += (int32_t)pSrcA[(m + 0) * N + n] * pSrcB[(O - 1) * N + n];
        }

        sum0 = alpha * sum0 + beta * pSrcC[(m + 0) * O + (O - 1)];

        // Requantize value
        sum0 = (sum0 * mul[m + 0] + add[m + 0]) >> log2D;
        pDstY[(m + 0) * O + (O - 1)] = (int8_t)CLAMP(sum0, -128, 127);
      }
    }
  }
}

void RQGemm_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset, int8_t output_min, int8_t output_max) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);

  const int32_t rqs_bias = ((1 << (log2D - 1))) * rounding;
  const int32_t bias = beta * C_offset;

  int32_t _add = add[0];
  int32_t _mul = mul[0];

  if (transA == 0 && transB == 0) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  } else if (transA == 1 && transB == 0) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[n * P + p] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  } else if (transA == 0 && transB == 1) {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[m * N + n] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  } else {
    for (uint32_t m = core_id / c; m < M; m += numThreads / c) {
      if (per_row_quant) {
        _mul = mul[m];
        _add = add[m];
      }
      for (uint32_t p = c_start; p < c_end; ++p) {
        int32_t sum = 0;
        for (uint32_t n = 0; n < N; ++n) {
          sum += (int32_t)(pSrcA[n * M + m] + A_offset) *
                 (pSrcB[p * N + n] + B_offset);
        }
        // Requantize value
        sum = alpha * sum + beta * pSrcC[m * P + p] + bias;
        sum = sum * _mul + rqs_bias + _add;
        sum = (sum >> log2D) + Y_offset;
        pDstY[m * P + p] = (int8_t)CLAMP(sum, output_min, output_max);
      }
    }
  }
}

void RQGemm_offset_unrolled_2x2_parallel_s8_rv32im(
    int8_t const *__restrict__ pSrcA, int8_t const *__restrict__ pSrcB,
    int32_t const *__restrict__ pSrcC, int8_t *__restrict__ pDstY, uint32_t M,
    uint32_t N, uint32_t P, int32_t alpha, int32_t beta, int32_t transA,
    int32_t transB, int32_t *mul, int32_t *add, int32_t log2D, bool rounding,
    bool per_row_quant, int32_t A_offset, int32_t B_offset, int32_t C_offset,
    int32_t Y_offset) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();

  // Parallelize by assigning each core one row
  uint32_t const c = 1; // How many columns to split the matrix into
  uint32_t const c_start = (P / c) * (core_id % c);
  uint32_t const c_end = (P / c) * ((core_id % c) + 1);

  const int32_t rqs_bias = ((1 << (log2D - 1))) * rounding;
  const int32_t bias = beta * C_offset;

  int32_t _add0 = add[0];
  int32_t _add1 = add[0];
  int32_t _mul0 = mul[0];
  int32_t _mul1 = mul[0];

  if (transA == 0 && transB == 0) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(m + 0) * N + n + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(m + 0) * N + n + 1] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(m + 1) * N + n + 0] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(m + 1) * N + n + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(n + 0) * P + p + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(n + 0) * P + p + 1] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(n + 1) * P + p + 0] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(n + 1) * P + p + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  } else if (transA == 1 && transB == 0) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(n + 0) * M + m + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(n + 1) * M + m + 0] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(n + 0) * M + m + 1] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(n + 1) * M + m + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(n + 0) * P + p + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(n + 0) * P + p + 1] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(n + 1) * P + p + 0] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(n + 1) * P + p + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  } else if (transA == 0 && transB == 1) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(m + 0) * N + n + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(m + 0) * N + n + 1] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(m + 1) * N + n + 0] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(m + 1) * N + n + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(p + 0) * N + n + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(p + 1) * N + n + 0] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(p + 0) * N + n + 1] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(p + 1) * N + n + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  } else if (transA == 1 && transB == 1) {
    for (uint32_t m = 2 * (core_id / c); m < M; m += 2 * (numThreads / c)) {
      if (per_row_quant) {
        _mul0 = mul[m + 0];
        _mul1 = mul[m + 1];
        _add0 = add[m + 0];
        _add1 = add[m + 1];
      }
      for (uint32_t p = c_start; p < c_end; p += 2) {
        int32_t c00 = 0;
        int32_t c01 = 0;
        int32_t c10 = 0;
        int32_t c11 = 0;
        for (uint32_t n = 0; n < N; n += 2) {
          // Explicitly load the values first to help with scheduling
          int8_t val_a00 = (int8_t)(pSrcA[(n + 0) * M + m + 0] + A_offset);
          int8_t val_a01 = (int8_t)(pSrcA[(n + 1) * M + m + 0] + A_offset);
          int8_t val_a10 = (int8_t)(pSrcA[(n + 0) * M + m + 1] + A_offset);
          int8_t val_a11 = (int8_t)(pSrcA[(n + 1) * M + m + 1] + A_offset);
          int8_t val_b00 = (int8_t)(pSrcB[(p + 0) * N + n + 0] + B_offset);
          int8_t val_b01 = (int8_t)(pSrcB[(p + 1) * N + n + 0] + B_offset);
          int8_t val_b10 = (int8_t)(pSrcB[(p + 0) * N + n + 1] + B_offset);
          int8_t val_b11 = (int8_t)(pSrcB[(p + 1) * N + n + 1] + B_offset);
          c00 += val_a00 * val_b00;
          c00 += val_a01 * val_b10;
          c01 += val_a00 * val_b01;
          c01 += val_a01 * val_b11;
          c10 += val_a10 * val_b00;
          c10 += val_a11 * val_b10;
          c11 += val_a10 * val_b01;
          c11 += val_a11 * val_b11;
        }

        c00 = c00 * alpha + beta * pSrcC[(m + 0) * P + p + 0] + bias;
        c01 = c01 * alpha + beta * pSrcC[(m + 0) * P + p + 1] + bias;
        c10 = c10 * alpha + beta * pSrcC[(m + 1) * P + p + 0] + bias;
        c11 = c11 * alpha + beta * pSrcC[(m + 1) * P + p + 1] + bias;

        c00 = c00 * _mul0 + rqs_bias + _add0;
        c01 = c01 * _mul0 + rqs_bias + _add0;
        c10 = c10 * _mul1 + rqs_bias + _add1;
        c11 = c11 * _mul1 + rqs_bias + _add1;

        c00 = (c00 >> log2D) + Y_offset;
        c01 = (c01 >> log2D) + Y_offset;
        c10 = (c10 >> log2D) + Y_offset;
        c11 = (c11 >> log2D) + Y_offset;

        pDstY[(m + 0) * P + p + 0] = (int8_t)CLAMP(c00, -128, 127);
        pDstY[(m + 0) * P + p + 1] = (int8_t)CLAMP(c01, -128, 127);
        pDstY[(m + 1) * P + p + 0] = (int8_t)CLAMP(c10, -128, 127);
        pDstY[(m + 1) * P + p + 1] = (int8_t)CLAMP(c11, -128, 127);
      }
    }
  }
}
