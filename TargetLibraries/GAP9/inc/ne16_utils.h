/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * NE16 utility kernels for GAP9
 */

#ifndef __NE16_UTILS_GAP9__
#define __NE16_UTILS_GAP9__

#include "pmsis.h"
#include "CNN_BasicKernels_fp32.h"

typedef struct {
    int8_t *In;
    uint8_t *Out;
    int size;
} ne16_int8_to_uint8_T;

/* Multi-core SIMD int8 → uint8 conversion (+128 offset) */
void ne16_int8_to_uint8(ne16_int8_to_uint8_T *Arg);

#endif
