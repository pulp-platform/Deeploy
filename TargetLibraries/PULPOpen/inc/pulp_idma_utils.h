/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich, University of Bologna and
 * Fondazione Chips-IT
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __PULP_IDMA_UTILS_H__
#define __PULP_IDMA_UTILS_H__

#include "pmsis.h"

// HAL length is unsigned short: split long 1D transfers at runtime, so every
// tile shares one call shape. Word-aligned chunks keep the offsets aligned.
#define DEEPLOY_IDMA_CHUNK (0xFFFFu & ~3u)

static inline void deeploy_idma_transfer_1d_and_wait(unsigned int direction,
                                                     unsigned int ext,
                                                     unsigned int loc,
                                                     unsigned int size) {
  while (size > 0xFFFFu) {
    pulp_idma_transfer_1d_and_wait(direction, ext, loc, DEEPLOY_IDMA_CHUNK);
    ext += DEEPLOY_IDMA_CHUNK;
    loc += DEEPLOY_IDMA_CHUNK;
    size -= DEEPLOY_IDMA_CHUNK;
  }
  pulp_idma_transfer_1d_and_wait(direction, ext, loc, (unsigned short)size);
}

// Asynchronous variants: start the transfer and return. Completion is awaited
// per direction (the iDMA keeps one transfer ID counter per direction), with
// deeploy_idma_wait_channel_id_input/output, where the data is needed.
static inline void deeploy_idma_transfer_1d(unsigned int direction,
                                            unsigned int ext, unsigned int loc,
                                            unsigned int size) {
  while (size > 0) {
    unsigned int len = size > 0xFFFFu ? DEEPLOY_IDMA_CHUNK : size;
    if (direction == 1) {
      pulp_cl_idma_L2ToL1(ext, loc, (unsigned short)len);
    } else {
      pulp_cl_idma_L1ToL2(loc, ext, (unsigned short)len);
    }
    ext += len;
    loc += len;
    size -= len;
  }
}

static inline void deeploy_idma_transfer_2d(unsigned int direction,
                                            unsigned int ext, unsigned int loc,
                                            unsigned short size,
                                            unsigned int stride_ext,
                                            unsigned int stride_loc,
                                            unsigned int num_reps) {
  if (direction == 1) {
    pulp_cl_idma_L2ToL1_2d(ext, loc, size, stride_ext, stride_loc, num_reps);
  } else {
    pulp_cl_idma_L1ToL2_2d(loc, ext, size, stride_loc, stride_ext, num_reps);
  }
}

// Rows of `size` bytes, `num_reps` rows per plane, `num_reps_3d` planes.
// stride_*_3d are plane-to-plane strides, but the iDMA (idma_nd_midend) adds
// the 3D stride to the address of the last row of a plane, not to its start:
// convert them. Unsigned wrap-around handles negative results.
static inline void
deeploy_idma_transfer_3d(unsigned int direction, unsigned int ext,
                         unsigned int loc, unsigned short size,
                         unsigned int stride_ext, unsigned int stride_loc,
                         unsigned int num_reps, unsigned int stride_ext_3d,
                         unsigned int stride_loc_3d, unsigned int num_reps_3d) {
  unsigned int last_row = num_reps - 1;
  stride_ext_3d -= last_row * stride_ext;
  stride_loc_3d -= last_row * stride_loc;
  if (direction == 1) {
    pulp_cl_idma_L2ToL1_3d(ext, loc, size, stride_ext, stride_loc, num_reps,
                           stride_ext_3d, stride_loc_3d, num_reps_3d);
  } else {
    pulp_cl_idma_L1ToL2_3d(loc, ext, size, stride_loc, stride_ext, num_reps,
                           stride_loc_3d, stride_ext_3d, num_reps_3d);
  }
}

static inline void deeploy_idma_wait_channel_id_input(void) {
  plp_cl_dma_barrier_toL1();
}

static inline void deeploy_idma_wait_channel_id_output(void) {
  plp_cl_dma_barrier_toL2();
}

#endif // __PULP_IDMA_UTILS_H__
