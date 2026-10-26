/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Concat.h"

void Carfield_Concat2(float32_t *data_in_1, float32_t *data_in_2, float32_t *data_out,
                      int32_t size_1, int32_t size_2,
                      int32_t core_id, int32_t numThreads) {

    int32_t size = size_1 + size_2;

    int32_t chunk = (size + numThreads - 1) / numThreads;
    int32_t start = core_id * chunk;
    int32_t end   = start + chunk;
    if (end > size) end = size;
    if (start > size) start = size;

    int32_t a0 = start < size_1 ? start : size_1;
    int32_t a1 = end   < size_1 ? end   : size_1;
    for (int32_t i = a0; i < a1; i++) {
        data_out[i] = data_in_1[i];
    }

    int32_t b0 = start > size_1 ? start - size_1 : 0;
    int32_t b1 = end   > size_1 ? end   - size_1 : 0;
    for (int32_t i = b0; i < b1; i++) {
        data_out[size_1 + i] = data_in_2[i];
    }
}
