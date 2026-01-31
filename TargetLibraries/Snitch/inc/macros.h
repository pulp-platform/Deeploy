/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_MACROS_HEADER_
#define __DEEPLOY_MATH_MACROS_HEADER_

#define INT_LOG2(x) __builtin_ctz(x)

#ifndef CLAMP
#define CLAMP(x, low, high)                                                    \
  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))
#endif

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef MAX
#define MAX(a, b) ((a) > (b) ? (a) : (b))
#endif

// JUNGVI: The following macros are here to ensure compatibility with some
// PULP-NN kernels
#define clips8(x) CLAMP(x, -128, 127)

#endif //__DEEPLOY_MATH_MACROS_HEADER_
