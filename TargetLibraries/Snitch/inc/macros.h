/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MACROS_HEADER_
#define __DEEPLOY_MATH_MACROS_HEADER_

#define INT_LOG2(x) __builtin_ctz(x)
#define CLAMP(x, low, high)                                                    \
  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// JUNGVI: The following macros are here to ensure compatibility with some
// PULP-NN kernels
#define clips8(x) CLAMP(x, -128, 127)

#endif //__DEEPLOY_MATH_MACROS_HEADER_
