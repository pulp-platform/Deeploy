/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Utils.h"
#include "Types.h"
#include "pulp.h"
#include <stddef.h>
#include <stdio.h>

#define DDR_ARENA_BASE                                                         \
  0x82600000 // IT MUST BE CHANGED EVERY TIME BASED ON WHAT IT IS ALLOCATED
             // BEFORE THE DDR ARENA. THE DDR ARENA ADDRESS CAN BE READ FORM
             // LAYOUT.TXT AFTER NETWORK GENERATION

// Top of the DDR window
#define DDR_WINDOW_END 0x100000000ULL

static float32_t *ddr_buffer_ptr = (float32_t *)DDR_ARENA_BASE;

uint32_t bytes_left = (uint32_t)(DDR_WINDOW_END - DDR_ARENA_BASE);

#define DDR_FREE_SLOTS 128

typedef struct {
  uint32_t addr;
  uint32_t size;
} ddr_block_t;

static ddr_block_t ddr_free_list[DDR_FREE_SLOTS];
static uint32_t ddr_free_used = 0;

uint32_t bytes_recycled = 0;

uint32_t bytes_peak = 0;

void *deeploy_malloc(const size_t size) {

  uint32_t round = (((size + 63) / 64) * 64); // Rounding to 64 bytes.

  for (uint32_t i = 0; i < ddr_free_used; i++) {
    if (ddr_free_list[i].size >= round) {
      uint32_t addr = ddr_free_list[i].addr;
      uint32_t rest = ddr_free_list[i].size - round;

      if (rest == 0) {
        ddr_free_list[i] = ddr_free_list[--ddr_free_used];
      } else {
        ddr_free_list[i].addr += round;
        ddr_free_list[i].size = rest;
      }
      bytes_recycled -= round;
      return (void *)addr;
    }
  }

  if (round > bytes_left) {
    printf("deeploy_malloc: out of DDR arena, requested %u bytes, %u left, %u "
           "recycled\n\r",
           (unsigned)size, (unsigned)bytes_left, (unsigned)bytes_recycled);
    hal_cluster_ctrl_return_set_remote(hal_cluster_id(), 0x7FFFFFFF);
    hal_cluster_ctrl_eoc_set_remote(hal_cluster_id(), 1);
    while (1)
      ;
  }

  float32_t *ret = ddr_buffer_ptr;

  ddr_buffer_ptr += round / 4;

  bytes_left -= round;
  if ((uint32_t)ddr_buffer_ptr - DDR_ARENA_BASE > bytes_peak) {
    bytes_peak = (uint32_t)ddr_buffer_ptr -
                 DDR_ARENA_BASE; // To know how much margin we have.
  }
  return ret;
}

void deeploy_free(void *const ptr, const size_t size) {

  if (ptr == NULL) {
    return;
  }

  uint32_t addr = (uint32_t)ptr;
  uint32_t round = (((size + 63) / 64) * 64);

  for (uint32_t i = 0; i < ddr_free_used;) {
    if (ddr_free_list[i].addr + ddr_free_list[i].size == addr) {
      addr = ddr_free_list[i].addr;
      round += ddr_free_list[i].size;
      bytes_recycled -= ddr_free_list[i].size;
      ddr_free_list[i] = ddr_free_list[--ddr_free_used];
      i = 0;
      continue;
    }
    if (addr + round == ddr_free_list[i].addr) {
      round += ddr_free_list[i].size;
      bytes_recycled -= ddr_free_list[i].size;
      ddr_free_list[i] = ddr_free_list[--ddr_free_used];
      i = 0;
      continue;
    }
    i++;
  }

  if (addr + round == (uint32_t)ddr_buffer_ptr) {
    ddr_buffer_ptr = (float32_t *)addr;
    bytes_left += round;
    return;
  }

  if (ddr_free_used == DDR_FREE_SLOTS) {
    printf("deeploy_free: free list full (%u slots), leaking %u bytes at "
           "0x%08x\n\r",
           (unsigned)DDR_FREE_SLOTS, (unsigned)round, (unsigned)addr);
    return;
  }

  ddr_free_list[ddr_free_used].addr = addr;
  ddr_free_list[ddr_free_used].size = round;
  ddr_free_used++;
  bytes_recycled += round;
}
