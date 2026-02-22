/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
// #include "perf_utils.h"

void SILU_s8_s32(int8_t *data_in, int32_t *data_out, int32_t dataSize,
                 int32_t input_offset) {

    // int8_t core_id = pi_core_id();
    // int8_t log2Core = LOG2(NUM_CORES);

    //RW: Performance monitoring is currently disabled 
    // perf_stats_t perf_start, perf_end, perf_total;

    // Initialize and start performance counters (only core 0)
    // if (core_id == 0) {
    //   perf_bench_init();
    //   perf_bench_start();
    //perf_bench_read(&perf_start);
    // }

    for (int i = 0; i < dataSize; i++) {
      int32_t x = data_in[i] + 128 - input_offset;
      data_out[i] = SILU_lut_s8_s32[x];
     }

    // RW: Stop performance counters and print results (only core 0)
    // if (core_id == 0) {
    //  perf_bench_stop();
    //  perf_bench_read(&perf_end);
    //  perf_bench_diff(&perf_total, &perf_end, &perf_start);

    //  char label[100];
    //  snprintf(label, sizeof(label), "GEMM M=%u N=%u O=%u transA=%u transB=%u",
    //           M, N, O, transA, transB);
    //  perf_bench_print(label, &perf_total);
  // }
}