/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Add.h"

void Carfield_Add(float32_t *data_in_1, float32_t *data_in_2,
                  float32_t *data_out, int32_t size, int32_t core_id, int32_t numThreads) {

    int32_t chunk = (size + numThreads - 1) / numThreads;
    int32_t start = core_id * chunk;
    int32_t end   = start + chunk;
    if (end > size) end = size;          // Last core can have less work to do
    if (start > size) start = size;      // With size < numThreads, some cores will have no work to do

    for (int32_t i = start; i < end; i++)
        data_out[i] = data_in_1[i] + data_in_2[i];

}
