/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tile.h"

void MAGIA_Add(int8_t *pIn1, int8_t *pIn2, int32_t *pOut, uint32_t size, int32_t offset) {

    uint32_t hartid = get_hartid();
    uint32_t tile_size = (size + NUM_HARTS - 1) / NUM_HARTS;
    uint32_t start = tile_size * hartid;
    uint32_t stop = tile_size * (hartid + 1);

    if (stop > size) {
        stop = size;
    }

    for (uint32_t i = start; i < stop; i++) {
        pOut[i] = (int32_t)(pIn1[i]) + (int32_t)(pIn2[i]) + offset;
    }
}
