/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CYCLECOUNTER
#define CYCLECOUNTER

extern volatile unsigned int *DWT_CYCCNT;
extern volatile unsigned int *DWT_CONTROL;
extern volatile unsigned int *SCB_DEMCR;

// Resets the internal cycle counter to zero
void ResetTimer(void);

// Starts the internal cycle counter
void StartTimer(void);

// Stops the internal cycle counter
void StopTimer(void);

// Returns the current number of cycles according to the internal cycle counter
unsigned int getCycles(void);

#endif
