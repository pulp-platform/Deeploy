/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CycleCounter.h"
#include "pmsis.h"

void ResetTimer() {
  pi_perf_conf(PI_PERF_CYCLES);
  pi_perf_cl_reset();
}

void StartTimer() { pi_perf_cl_start(); }

void StopTimer() { pi_perf_cl_stop(); }

unsigned int getCycles() { return pi_perf_cl_read(PI_PERF_CYCLES); }
