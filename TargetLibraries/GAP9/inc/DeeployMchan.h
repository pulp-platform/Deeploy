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
 * This header configures and includes mchan.h with proper GAP9-specific settings.
 * Based on DORY's GAP9 DMA implementation.
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
#define MCHAN_EVENT_BIT (CLUSTER_IRQ_DMA0)  // Typically 8
#endif
#endif

// Now include the mchan.h header with all configurations set
#include "mchan.h"

#endif // _DEEPLOY_MCHAN_H
