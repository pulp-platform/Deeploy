/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeploySnitchMath.h"

// Define ENABLE_INSTR_COUNTER to enable instruction counting (causes warnings
// in gvsoc) #define ENABLE_INSTR_COUNTER

static uint32_t timer_init[NUM_CORES] __attribute__((section(".l1")));
static uint32_t timer_end[NUM_CORES] __attribute__((section(".l1")));
#ifdef ENABLE_INSTR_COUNTER
static uint32_t instr_init[NUM_CORES] __attribute__((section(".l1")));
static uint32_t instr_end[NUM_CORES] __attribute__((section(".l1")));
#endif

static uint32_t running[NUM_CORES] __attribute__((section(".l1")));

void ResetTimer() {
  snrt_reset_perf_counter(SNRT_PERF_CNT0);
  uint32_t const core_id = snrt_global_core_idx();
  uint32_t _timer_init = read_csr(mcycle);
  timer_init[core_id] = _timer_init;
  timer_end[core_id] = _timer_init;
#ifdef ENABLE_INSTR_COUNTER
  uint32_t _instr_init = read_csr(minstret);
  instr_init[core_id] = _instr_init;
  instr_end[core_id] = _instr_init;
#endif
  running[core_id] = 0;
}

void StartTimer() {
  if (snrt_is_dm_core()) {
    snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_CYCLES, 0);
  }
  uint32_t const core_id = snrt_global_core_idx();
  timer_init[core_id] = read_csr(mcycle);
#ifdef ENABLE_INSTR_COUNTER
  instr_init[core_id] = read_csr(minstret);
#endif
  running[core_id] = 1;
}

void StopTimer() {
  if (snrt_is_dm_core()) {
    snrt_stop_perf_counter(SNRT_PERF_CNT0);
  }
  uint32_t const core_id = snrt_global_core_idx();
  timer_end[core_id] = read_csr(mcycle);
#ifdef ENABLE_INSTR_COUNTER
  instr_end[core_id] = read_csr(minstret);
#endif
  running[core_id] = 0;
}

uint32_t getCycles() {
  uint32_t const core_id = snrt_global_core_idx();
  if (running[core_id]) {
    return read_csr(mcycle) - timer_init[core_id];
  } else {
    return timer_end[core_id] - timer_init[core_id];
  }
}

uint32_t getInstr(void) {
#ifdef ENABLE_INSTR_COUNTER
  uint32_t const core_id = snrt_global_core_idx();

  if (running[core_id]) {
    return read_csr(minstret) - instr_init[core_id];
  } else {
    return instr_end[core_id] - instr_init[core_id];
  }
#else
  return 0; // Instruction counting disabled
#endif
}