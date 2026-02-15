/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef _DORY_DMA_H
#define _DORY_DMA_H

typedef struct {
  void *ext;
  void *loc;
  unsigned short hwc_to_chw;
  unsigned short stride_2d;
  unsigned short number_of_2d_copies;
  unsigned short stride_1d;
  unsigned short number_of_1d_copies;
  unsigned int length_1d_copy;
  unsigned int mchan_cmd;
  int dir; // 0 l1->l2, 1 l2->l1
  int tid;
} DMA_copy;

void dory_dma_memcpy_hwc_to_chw(DMA_copy *copy);

void dory_dma_memcpy_1d_async(DMA_copy *copy);

void dory_dma_memcpy_2d_async(DMA_copy *copy);

void dory_dma_memcpy_3d_async(DMA_copy *copy);

void dory_dma_memcpy_async(DMA_copy *copy);

void dory_dma_memcpy_mindims_async(DMA_copy *copy);

void dory_dma_free(DMA_copy *copy);

void dory_dma_barrier(DMA_copy *copy);

int dory_dma_allocate();
#endif
