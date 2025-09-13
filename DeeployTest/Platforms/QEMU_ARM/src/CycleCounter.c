/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "CycleCounter.h"

volatile unsigned int *DWT_CYCCNT =
    (unsigned int *)0xE0001004; // address of the register
volatile unsigned int *DWT_CONTROL =
    (unsigned int *)0xE0001000; // address of the register
volatile unsigned int *SCB_DEMCR =
    (unsigned int *)0xE000EDFC; // address of the register

static unsigned int prev_val = 0;
static int stopped = 0;

void ResetTimer() {

  *SCB_DEMCR = *SCB_DEMCR | 0x01000000;
  *DWT_CYCCNT = 0; // reset the counter
  *DWT_CONTROL = 1;
  stopped = 1;
  prev_val = 0;
}

void StartTimer() {
  prev_val = *DWT_CYCCNT;
  stopped = 0;
}

void StopTimer() {
  prev_val = *DWT_CYCCNT - prev_val;
  stopped = 1;
}

unsigned int getCycles() {
  if (stopped) {
    return prev_val;
  } else {
    return *DWT_CYCCNT - prev_val;
  }
}
