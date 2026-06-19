/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _DEEPLOY_MCHAN_H
#define _DEEPLOY_MCHAN_H

/*
 * GAP9 MCHAN v7 configuration wrapper for Deeploy
 *
 * This header configures and includes mchan.h with proper GAP9-specific
 * settings. Based on DORY's GAP9 DMA implementation.
 */

#include "pmsis.h"

// Define MCHAN base address if not already defined
#ifndef MCHAN_BASE_ADDR
#define MCHAN_BASE_ADDR (CLUSTER_PERIPHERALS_ADDR + CLUSTER_MCHAN_OFFSET)
#endif

// Define MCHAN version (GAP9 uses v7)
#ifndef MCHAN_VERSION
#define MCHAN_VERSION 7
#endif

// Use event-based synchronization (recommended for GAP9)
#ifndef MCHAN_POLLED
#define MCHAN_EVENT
#endif

// Define event bit for cluster DMA
#ifdef MCHAN_EVENT
#ifndef MCHAN_EVENT_BIT
#define MCHAN_EVENT_BIT (CLUSTER_IRQ_DMA0) // Typically 8
#endif
#endif

// Now include the mchan.h header with all configurations set
#include "mchan.h"

// ---------------------------------------------------------------------------
// MCHAN v7 channel API for PULPOpen-style generated tiling code
// Provides mchan_transfer_1d / mchan_channel_alloc/wait/free interface
// ---------------------------------------------------------------------------
#ifndef MCHAN_CHANNEL_ID_MAX
#define MCHAN_CHANNEL_ID_MAX (15)
#endif

static volatile uint32_t *const cmd_ptr =
    (volatile uint32_t *const)(MCHAN_BASE_ADDR + 0x0);
static volatile uint32_t *const status_ptr =
    (volatile uint32_t *const)(MCHAN_BASE_ADDR + 0x4);

static inline void mchan_transfer_1d(uint32_t cmd, void *loc, void *ext) {
  *cmd_ptr = cmd;
  *cmd_ptr = (uint32_t)loc;
  *cmd_ptr = (uint32_t)ext;
}

static inline uint32_t mchan_channel_alloc() { return *cmd_ptr; }

static inline void mchan_channel_free(uint32_t channel_id) {
  *status_ptr = 1u << channel_id;
}

static inline uint32_t mchan_channel_is_busy(uint32_t channel_id) {
  return *status_ptr & (1u << channel_id);
}

static inline void mchan_channel_wait(uint32_t channel_id) {
#if defined(MCHAN_EVENT)
  while (mchan_channel_is_busy(channel_id))
    eu_evt_maskWaitAndClr(1u << MCHAN_EVENT_BIT);
#else
  while (mchan_channel_is_busy(channel_id))
    ;
#endif
}

#endif // _DEEPLOY_MCHAN_H
