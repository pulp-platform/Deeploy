/* =====================================================================
 * Title:        mchan_v6.h
 * Description:
 *
 * $Date:        26.07.2024
 *
 * ===================================================================== */
/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.

 * Adopted from PULP-SDK (https://github.com/pulp-platform/pulp-sdk), released
 under Apache 2.0

 */

#ifndef __MCHAN_V6_H__
#define __MCHAN_V6_H__

// Requires to have MCHAN_BASE_ADDR, MCHAN_EVENT defined outside of header
#ifndef MCHAN_BASE_ADDR
#error "[mchan_v6.h] MCHAN_BASE_ADDR not defined!"
#endif

#if !defined(MCHAN_EVENT) && !defined(MCHAN_POLLED)
#error "[mchan_v6.h] Nor MCHAN_EVENT nor MCHAN_POLLED defined!"
#endif

#if defined(MCHAN_EVENT) && !defined(MCHAN_EVENT_BIT)
#error                                                                         \
    "[mchan_v6.h] MCHAN_EVENT_BIT should be defined when using events as signalization!"
#endif

#if !defined(MCHAN_VERSION)
#define MCHAN_VERSION 6
#elif MCHAN_VERSION != 6
#error "[mchan_v6.h] Illegal MCHAN_VERSION. Supported only 6"
#endif

#include "pmsis.h"

#define MCHAN_TRANSFER_LEN_SIZE (16)

#define MCHAN_CMD_FLAG_DIRECTION_LOC2EXT (0 << (MCHAN_TRANSFER_LEN_SIZE + 0))
#define MCHAN_CMD_FLAG_DIRECTION_EXT2LOC (1 << (MCHAN_TRANSFER_LEN_SIZE + 0))
#define MCHAN_CMD_FLAG_INCREMENTAL (1 << (MCHAN_TRANSFER_LEN_SIZE + 1))
#define MCHAN_CMD_FLAG_2D_TRANSFER_EXTERNAL (1 << (MCHAN_TRANSFER_LEN_SIZE + 2))
#define MCHAN_CMD_FLAG_EVENT_ENABLE (1 << (MCHAN_TRANSFER_LEN_SIZE + 3))
#define MCHAN_CMD_FLAG_INTERRUPT_ENABLE (1 << (MCHAN_TRANSFER_LEN_SIZE + 4))
#define MCHAN_CMD_FLAG_BROADCAST_FINISH (1 << (MCHAN_TRANSFER_LEN_SIZE + 5))

static volatile uint32_t *const cmd_ptr =
    (volatile uint32_t *const)(MCHAN_BASE_ADDR + 0x0);
static volatile uint32_t *const status_ptr =
    (volatile uint32_t *const)(MCHAN_BASE_ADDR + 0x4);

static void mchan_transfer_1d(uint32_t cmd, void *loc, void *ext) {
  // TODO: assert flags are set correctly
  *cmd_ptr = (uint32_t)cmd;
  *cmd_ptr = (uint32_t)loc;
  *cmd_ptr = (uint32_t)ext;
}

static void mchan_transfer_2d_ext_strided(uint32_t cmd, void *loc, void *ext,
                                          uint16_t ext_size_1d,
                                          uint16_t ext_stride_2d) {
  // TODO: assert flags are set correctly
  *cmd_ptr = (uint32_t)cmd;
  *cmd_ptr = (uint32_t)loc;
  *cmd_ptr = (uint32_t)ext;
  *cmd_ptr = (uint32_t)ext_size_1d | ((uint32_t)ext_stride_2d << 16);
}

static uint32_t mchan_channel_alloc() { return *cmd_ptr; }

static void mchan_channel_free(uint32_t channel_id) {
  // TODO: assert channel_id is smaller then 32
  *status_ptr = 1 << channel_id;
}

static uint32_t mchan_channel_is_busy(uint32_t channel_id) {
  // TODO: assert channel_id is smaller then 32
  return *status_ptr & (1 << channel_id);
}

static void mchan_channel_wait(uint32_t channel_id) {
  // TODO: assert channel_id is smaller then 32
#if defined(MCHAN_EVENT)
  while (mchan_channel_is_busy(channel_id))
    eu_evt_maskWaitAndClr(1 << MCHAN_EVENT_BIT);
#elif defined(MCHAN_POLLED)
  while (mchan_channel_is_busy(channel_id))
    ;
#endif
}

#endif // __MCHAN_V6_H__
