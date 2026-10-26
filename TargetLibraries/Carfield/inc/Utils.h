/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __CARFIELD_UTILS_H_
#define __CARFIELD_UTILS_H_

#define BEGIN_SINGLE_CORE if (core_id == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (core_id == 0)

#include <stddef.h>  // for size_t
#include <stdint.h>  // for uint32_t
#include "pulp.h"

void *deeploy_malloc(const size_t size);
void  deeploy_free(void *const ptr, const size_t size);

// useful PULP cluster heap from 0x50004000 to 0x50020000, since each of 8 cores
// has 2KiB for stack, in total first 16Kib from 0x50000000
#define CARFIELD_L1_END 0x50020000U

// Bytes still available in the DDR arena: the tail never handed out, so
// (arena size - bytes_left) is the peak the bump pointer ever reached.
extern uint32_t bytes_left;

// Bytes sitting in the free list, waiting to be reused.
extern uint32_t bytes_recycled;

// Peak of the DDR arena: the smallest arena the program would run in.
extern uint32_t bytes_peak;

#endif // __CARFIELD_UTILS_H_
