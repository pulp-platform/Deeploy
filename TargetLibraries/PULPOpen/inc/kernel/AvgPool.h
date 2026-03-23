/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_AVGPOOL_KERNEL_HEADER_
#define __DEEPLOY_MATH_AVGPOOL_KERNEL_HEADER_

#include "DeeployPULPMath.h"

void PULP_AvgPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                  uint32_t W, uint32_t H, uint32_t C,
                                  uint32_t Q, uint32_t P, uint32_t SQ,
                                  uint32_t SP, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right);


void PULP_AvgPool2d_fp32_fp32_CHW(const float32_t *__restrict__ pSrcA,
                                  uint32_t C, uint32_t H, uint32_t W,
                                  uint32_t P, uint32_t Q, uint32_t SP,
                                  uint32_t SQ, float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right);

/**
 * @brief Global Average Pooling forward pass (NCHW layout).
 *
 * For each (n, c), computes the mean over all (h, w) spatial positions:
 *   output[n*C + c] = sum_{h,w}(input[(n*C+c)*H*W + h*W + w]) / (H*W)
 *
 * Parallelized over channels: each core handles a contiguous chunk of channels.
 *
 * @param input   Input tensor  [N, C, H, W]  NCHW float32
 * @param output  Output tensor [N, C, 1, 1]  stored as [N*C] float32
 * @param N       Batch size
 * @param C       Number of channels
 * @param H       Spatial height
 * @param W       Spatial width
 */
void PULP_GlobalAveragePool_fp32(const float32_t *input, float32_t *output,
                                  uint32_t N, uint32_t C, uint32_t H, uint32_t W);

/**
 * @brief Global Average Pooling backward pass (NCHW layout).
 *
 * Distributes the upstream gradient evenly across all spatial positions:
 *   dX[n,c,h,w] = dY[n*C + c] / (H*W)
 *
 * Parallelized over channels: each core handles a contiguous chunk of channels.
 *
 * @param dY  Upstream gradient [N, C, 1, 1]  stored as [N*C] float32
 * @param dX  Gradient w.r.t. input [N, C, H, W] NCHW float32
 * @param N   Batch size
 * @param C   Number of channels
 * @param H   Spatial height
 * @param W   Spatial width
 */
void PULP_GlobalAveragePoolGrad_fp32(const float32_t *dY, float32_t *dX,
                                      uint32_t N, uint32_t C, uint32_t H, uint32_t W);

#endif // __DEEPLOY_MATH_AVGPOOL_KERNEL_HEADER_
