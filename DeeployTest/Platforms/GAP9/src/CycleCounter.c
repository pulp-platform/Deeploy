/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CycleCounter.h"
#include "pmsis.h"

void ResetTimer() {
  pi_perf_conf(1 << PI_PERF_CYCLES);
  pi_perf_reset();
}

void StartTimer() { pi_perf_start(); }

void StopTimer() { pi_perf_stop(); }

unsigned int getCycles() { return pi_perf_read(PI_PERF_CYCLES); }
