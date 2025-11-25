/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_TYPES_HEADER_
#define __DEEPLOY_BASIC_MATH_TYPES_HEADER_

// generic floating point types
typedef double float64_t;
typedef float float32_t;
// Note: float16_t uses _Float16 (C23) or compiler extensions
// For generic platforms without FP16 hardware support, we use float32_t as
// fallback
#if defined(__FLT16_MANT_DIG__) || defined(__ARM_FP16_FORMAT_IEEE)
typedef _Float16 float16_t;
#else
typedef float
    float16_t; // Fallback to float32 for platforms without FP16 support
#endif

#endif //__DEEPLOY_BASIC_MATH_TYPES_HEADER_
