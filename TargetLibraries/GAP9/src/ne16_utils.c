/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 * SPDX-License-Identifier: Apache-2.0
 *
 * NE16 utility kernels for GAP9
 */

#include "ne16_utils.h"

void ne16_int8_to_uint8(ne16_int8_to_uint8_T *Arg) {
    int8_t *In = Arg->In;
    uint8_t *Out = Arg->Out;
    int size = Arg->size;

    unsigned int CoreId = gap_coreid();
    unsigned int NCore = gap_ncore();
    unsigned int total_quads = size / 4;
    unsigned int Chunk = (total_quads + NCore - 1) / NCore;
    unsigned int First = Chunk * CoreId;
    unsigned int Last = First + Chunk;
    if (Last > total_quads) Last = total_quads;

    v4s offset = {-128, -128, -128, -128};
    for (unsigned int q = First; q < Last; q++) {
        *((v4s *)&Out[q * 4]) = *((v4s *)&In[q * 4]) + offset;
    }

    /* Handle remaining elements (size not multiple of 4) */
    if (CoreId == 0) {
        for (int i = total_quads * 4; i < size; i++) {
            Out[i] = (uint8_t)((int32_t)In[i] + 128);
        }
    }
}
