/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef CARFIELD_KERNEL_ADD_H  //used to prevent multiple inclusion of this header file
#define CARFIELD_KERNEL_ADD_H

#include <stdint.h>
#include "Types.h"

void Carfield_Add(
    float32_t *data_in_1,
    float32_t *data_in_2,
    float32_t *data_out,
    int32_t    size,
    int32_t core_id,
    int32_t numThreads
);

#endif // CARFIELD_KERNEL_ADD_H
