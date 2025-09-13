/*
 * Copyright (C) 2025, ETH Zurich and University of Bologna.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
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