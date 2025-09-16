/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_DMASTRUCT_HEADER_
#define __DEEPLOY_MATH_DMASTRUCT_HEADER_

#include "snrt.h"

typedef struct {
  void *dst;
  void *src;
  size_t size;
  size_t dst_stride;
  size_t src_stride;
  size_t repeat;
  snrt_dma_txid_t tid;
} DMA_copy;

#endif // __DEEPLOY_MATH_DMASTRUCT_HEADER_