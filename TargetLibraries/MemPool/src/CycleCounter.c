/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployMath.h"

static uint32_t timer_init[NUM_CORES] __attribute__((section(".l1")));
static uint32_t timer_end[NUM_CORES] __attribute__((section(".l1")));
static uint32_t instr_init[NUM_CORES] __attribute__((section(".l1")));
static uint32_t instr_end[NUM_CORES] __attribute__((section(".l1")));

void ResetTimer(void) {
  uint32_t const core_id = mempool_get_core_id();
  timer_init[core_id] = read_csr(mcycle);
  instr_init[core_id] = read_csr(minstret);
}

void StartTimer(void) {
  uint32_t const core_id = mempool_get_core_id();
  timer_init[core_id] = read_csr(mcycle);
  instr_init[core_id] = read_csr(minstret);
}

void StopTimer(void) {
  uint32_t const core_id = mempool_get_core_id();
  timer_end[core_id] = read_csr(mcycle);
  instr_end[core_id] = read_csr(minstret);
}

uint32_t getCycles(void) {
  uint32_t const core_id = mempool_get_core_id();
  return timer_end[core_id] - timer_init[core_id];
}

uint32_t getInstr(void) {
  uint32_t const core_id = mempool_get_core_id();
  return instr_end[core_id] - instr_init[core_id];
}