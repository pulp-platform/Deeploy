/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CARFIELD_KERNEL_ARGMAX_H
#define CARFIELD_KERNEL_ARGMAX_H

#include <stdint.h>
#include "Types.h"

void Carfield_ArgMaxChannel(
    float32_t *input,
    int64_t   *output,
    int32_t    C,
    int32_t    spatialSize,
    int32_t core_id,
    int32_t numThreads
);

#endif
