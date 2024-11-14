/*
 * ----------------------------------------------------------------------
 *
 * File: CycleCounter.c
 *
 * Last edited: 23.04.2024
 *
 * Copyright (C) 2024, ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
 *
 * ----------------------------------------------------------------------
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

static uint32_t timer_init[NUM_CORES] __attribute__((section(".l1")));
static uint32_t timer_end[NUM_CORES] __attribute__((section(".l1")));
static uint32_t instr_init[NUM_CORES] __attribute__((section(".l1")));
static uint32_t instr_end[NUM_CORES] __attribute__((section(".l1")));

static uint32_t running[NUM_CORES] __attribute__((section(".l1")));

void ResetTimer() {
  // snrt_reset_perf_counter(SNRT_PERF_CNT0);
  uint32_t const core_id = snrt_global_core_idx();
  uint32_t _timer_init = read_csr(mcycle);
  uint32_t _instr_init = read_csr(minstret);
  timer_init[core_id] = _timer_init;
  instr_init[core_id] = _instr_init;
  timer_end[core_id] = _timer_init;
  instr_end[core_id] = _instr_init;
  running[core_id] = 0;
}

void StartTimer() {
  // snrt_start_perf_counter(SNRT_PERF_CNT0, SNRT_PERF_CNT_CYCLES, 0);
  uint32_t const core_id = snrt_global_core_idx();
  timer_init[core_id] = read_csr(mcycle);
  instr_init[core_id] = read_csr(minstret);
  running[core_id] = 1;
}

void StopTimer() {
  // if (!snrt_is_dm_core()) {
  //   snrt_stop_perf_counter(SNRT_PERF_CNT0);
  // }
  uint32_t const core_id = snrt_global_core_idx();
  timer_end[core_id] = read_csr(mcycle);
  timer_end[core_id] = read_csr(minstret);
  running[core_id] = 0;
}

uint32_t getCycles() {
  // return snrt_get_perf_counter(SNRT_PERF_CNT0);
  uint32_t const core_id = snrt_global_core_idx();
  if (running[core_id]) {
    return read_csr(mcycle) - timer_init[core_id];
  } else {
    return timer_end[core_id] - timer_init[core_id];
  }
}

uint32_t getInstr(void) {
  uint32_t const core_id = snrt_global_core_idx();

  if (running[core_id]) {
    return read_csr(minstret) - instr_init[core_id];
  } else {
    return instr_end[core_id] - instr_init[core_id];
  }
}