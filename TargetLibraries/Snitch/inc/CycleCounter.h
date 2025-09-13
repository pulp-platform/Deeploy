/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_CYCLE_HEADER_
#define __DEEPLOY_MATH_CYCLE_HEADER_

#include <stdint.h>

// Resets the internal cycle and instruction counter to zero
void ResetTimer(void);

// Starts the internal cycle and instruction counter
void StartTimer(void);

// Stops the internal cycle and instruction counter
void StopTimer(void);

// Returns the current number of cycles according to the internal cycle counter
uint32_t getCycles(void);

// Returns the current number of instructions according to the internal
// instructions counter
uint32_t getInstr(void);

#endif //__DEEPLOY_MATH_CYCLE_HEADER_
