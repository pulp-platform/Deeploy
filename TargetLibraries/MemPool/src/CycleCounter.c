/* =====================================================================
 * Title:        CycleCounter.c
 * Description:
 *
 * Date:         06.12.2022
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except pSrcA compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to pSrcA writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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