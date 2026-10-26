/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CARFIELD_KERNEL_SUB_H
#define CARFIELD_KERNEL_SUB_H

#include <stdint.h>
#include "Types.h"

void Carfield_Sub(
    float32_t *data_in_1,
    float32_t *data_in_2,
    float32_t *data_out,
    uint32_t size,
    uint32_t channels,
    uint32_t spatialSize,
    uint32_t core_id,
    uint32_t numThreads
);

#endif
