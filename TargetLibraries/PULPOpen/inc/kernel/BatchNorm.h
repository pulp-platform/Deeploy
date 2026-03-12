/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_BATCHNORM_KERNEL_HEADER_
#define __DEEPLOY_MATH_BATCHNORM_KERNEL_HEADER_

#include "DeeployPULPMath.h"

/**
 * @brief Training-mode Batch Normalization forward pass (BatchNormInternal).
 *
 * Normalizes each channel over (N, H, W) using batch statistics.
 * Saves batch mean and 1/sqrt(var+eps) for use by the backward pass.
 * Running statistics are read but NOT updated (outputs 1,2 have no consumers).
 *
 * Parallelized over channels: each core handles a contiguous chunk of channels.
 *
 * @param X              Input  [N, C, H_in, W_in] NCHW float32
 * @param gamma          Affine scale [C]
 * @param beta           Affine bias  [C]
 * @param running_mean   EMA running mean [C] (read only)
 * @param running_var    EMA running variance [C] (read only)
 * @param Y              Output [N, C, H_in, W_in]
 * @param saved_mean     Stash: per-channel batch mean [C]
 * @param saved_inv_std  Stash: per-channel 1/sqrt(var+eps) [C]
 * @param N              Batch size
 * @param C              Number of channels
 * @param H_in           Spatial height
 * @param W_in           Spatial width
 * @param epsilon        Numerical stability term
 * @param momentum       EMA coefficient (currently unused — running stats not updated)
 */
void PULP_BatchNormInternal_fp32(const float32_t *X, const float32_t *gamma,
                                 const float32_t *beta, const float32_t *running_mean,
                                 const float32_t *running_var, float32_t *Y,
                                 float32_t *saved_mean, float32_t *saved_inv_std,
                                 uint32_t N, uint32_t C, uint32_t H_in, uint32_t W_in,
                                 float32_t epsilon, float32_t momentum);

/**
 * @brief Batch Normalization backward pass.
 *
 * Computes gradients w.r.t. input (dX), scale (dgamma), and bias (dbeta).
 * Uses the saved_mean / saved_inv_std stash from the forward pass.
 *
 * Standard BN backward formula (Ioffe & Szegedy, 2015):
 *   x_hat[n,c,h,w]  = (X[n,c,h,w] - saved_mean[c]) * saved_inv_std[c]
 *   dbeta[c]         = sum_{n,h,w} dY[n,c,h,w]
 *   dgamma[c]        = sum_{n,h,w} dY[n,c,h,w] * x_hat[n,c,h,w]
 *   dx_hat           = dY * gamma[c]
 *   dX[n,c,h,w]      = saved_inv_std[c] / N_total *
 *                       (N_total * dx_hat - dbeta[c] - x_hat * dgamma[c])
 *
 * Parallelized over channels: each core handles a contiguous chunk of channels.
 *
 * @param dY             Upstream gradient [N, C, H_in, W_in]
 * @param X              Original input from forward [N, C, H_in, W_in]
 * @param gamma          Affine scale [C]
 * @param saved_mean     Stash batch mean from forward [C]
 * @param saved_inv_std  Stash 1/sqrt(var+eps) from forward [C]
 * @param dX             Output: gradient w.r.t. input [N, C, H_in, W_in]
 * @param dgamma         Output: gradient w.r.t. scale [C]
 * @param dbeta          Output: gradient w.r.t. bias  [C]
 * @param N              Batch size
 * @param C              Number of channels
 * @param H_in           Spatial height
 * @param W_in           Spatial width
 * @param epsilon        Numerical stability term (unused — inv_std already computed)
 */
void PULP_BatchNormGrad_fp32(const float32_t *dY, const float32_t *X,
                              const float32_t *gamma, const float32_t *saved_mean,
                              const float32_t *saved_inv_std, float32_t *dX,
                              float32_t *dgamma, float32_t *dbeta, uint32_t N, uint32_t C,
                              uint32_t H_in, uint32_t W_in, float32_t epsilon);

#endif /* __DEEPLOY_MATH_BATCHNORM_KERNEL_HEADER_ */
