/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CARFIELD_KERNEL_CONCAT_H
#define CARFIELD_KERNEL_CONCAT_H

#include <stdint.h>
#include "Types.h"

void Carfield_Concat2(
    float32_t *data_in_1,
    float32_t *data_in_2,
    float32_t *data_out,
    int32_t    size_1,
    int32_t    size_2,
    int32_t core_id,
    int32_t numThreads
);

#endif
