/* =====================================================================
 * Title:        CycleCounter.c
 * Description:
 *
 * $Date:        26.07.2024
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
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
