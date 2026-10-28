/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_CHIMERA_MATH_HEADER_
#define __DEEPLOY_CHIMERA_MATH_HEADER_

// Define default empty wrapper for single core section
#ifndef BEGIN_SINGLE_CORE
#define BEGIN_SINGLE_CORE
#endif

#ifndef END_SINGLE_CORE
#define END_SINGLE_CORE
#endif

#ifndef SINGLE_CORE
#define SINGLE_CORE
#endif

#include <ctype.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// Memory island allocator, defined in chimera-sdk generic_runtime/src/alloc.c
// (the SDK provides no header for it).
void *memory_island_malloc(size_t size);
void memory_island_free(void *ptr);

#endif //__DEEPLOY_CHIMERA_MATH_HEADER_
