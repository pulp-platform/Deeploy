/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
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
