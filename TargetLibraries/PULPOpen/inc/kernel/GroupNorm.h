/*
 * SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_GROUPNORM_KERNEL_HEADER_
#define __DEEPLOY_MATH_GROUPNORM_KERNEL_HEADER_

#include "DeeployPULPMath.h"


/**
 * @brief Compute the gradient of GroupNorm with respect to the input X.
 *
 * GroupNorm: Y = gamma * (X - mean) / sqrt(var + eps) + beta
 *
 * dX = (1/std) * (gamma * dY - mean(gamma * dY) - X_norm * mean(gamma * dY * X_norm))
 * where X_norm = (X - mean) / std
 *
 * This kernel supports HW tiling and C parallelization.
 * Requires pre-computed grad_stat from GroupNormGradXStat_fp32_fp32.
 *
 * @param dY        Upstream gradient [N, C, H, W]
 * @param X         Original input [N, C, H, W]
 * @param gamma     Scale parameter [C]
 * @param mean      Pre-computed mean from forward [N, num_groups] (stride 2 in stat array)
 * @param inv_std   Pre-computed inverse std from forward [N, num_groups] (stride 2 in stat array)
 * @param grad_stat Pre-computed gradient statistics [N, num_groups, 2]
 * @param dX        Output gradient [N, C, H, W]
 * @param N         Batch size
 * @param C         Number of channels
 * @param H         Height (full, not tiled)
 * @param W         Width (full, not tiled)
 * @param num_groups Number of groups
 * @param start_c   Starting channel index for this core
 * @param end_c     Ending channel index for this core
 * @param start_hw  Starting HW index for this tile
 * @param end_hw    Ending HW index for this tile
 */
void GroupNormGradX_fp32_fp32(const float32_t *dY, const float32_t *X,
                               const float32_t *gamma, const float32_t *mean,
                               const float32_t *inv_std, const float32_t *grad_stat,
                               float32_t *dX,
                               int32_t N, int32_t C, int32_t H, int32_t W,
                               int32_t num_groups) ;

/**
 * @brief Compute the gradient of GroupNorm with respect to gamma (scale parameter).
 *
 * dGamma[c] = sum over (N, H, W) of (dY * X_norm)
 * where X_norm = (X - mean) / std
 *
 * @param dY        Upstream gradient [N, C, H, W]
 * @param X         Original input [N, C, H, W]
 * @param mean      Pre-computed mean [N, num_groups]
 * @param inv_std   Pre-computed inverse std [N, num_groups]
 * @param dGamma    Output gradient for gamma [C]
 * @param N         Batch size
 * @param C         Number of channels
 * @param H         Height
 * @param W         Width
 * @param num_groups Number of groups
 * @param epsilon   Epsilon for numerical stability
 * @param start_c   Starting channel index for this core
 * @param end_c     Ending channel index for this core
 */
void GroupNormGradW_fp32_fp32(const float32_t *dY, const float32_t *X,
                               const float32_t *mean, const float32_t *inv_std,
                               float32_t *dGamma,
                               int32_t N, int32_t C, int32_t H, int32_t W,
                               int32_t num_groups, float32_t epsilon,
                               int32_t start_c, int32_t end_c);

/**
 * @brief Compute the statistics (mean and inverse std) for GroupNorm.
 *
 * For each group, compute:
 *   mean = average of all elements in the group
 *   inv_std = 1 / sqrt(variance + epsilon)
 *
 * @param X         Input tensor [N, C, H, W]
 * @param stat      Output statistics [N, num_groups, 2] where [:,:,0]=mean, [:,:,1]=inv_std
 * @param N         Batch size
 * @param C         Number of channels
 * @param H         Height
 * @param W         Width
 * @param num_groups Number of groups
 * @param epsilon   Epsilon for numerical stability
 * @param start_group Starting group index for this core
 * @param end_group  Ending group index for this core
 */
void GroupNormalizationStat_fp32(const float32_t *X, float32_t *stat,
                                  int32_t N, int32_t C, int32_t H, int32_t W,
                                  int32_t num_groups, float32_t epsilon,
                                  int32_t start_group, int32_t end_group);

/**
 * @brief Apply GroupNorm normalization using pre-computed statistics.
 *
 * Y = gamma * (X - mean) * inv_std + beta
 *
 * @param X         Input tensor [N, C, H, W]
 * @param gamma     Scale parameter [C]
 * @param beta      Bias parameter [C]
 * @param stat      Pre-computed statistics [N, num_groups, 2] where [:,:,0]=mean, [:,:,1]=inv_std
 * @param Y         Output tensor [N, C, H, W]
 * @param N         Batch size
 * @param C         Number of channels
 * @param H         Height
 * @param W         Width
 * @param num_groups Number of groups
 * @param start_group Starting group index for this core
 * @param end_group  Ending group index for this core
 */
void GroupNormalization_fp32(const float32_t *X, const float32_t *gamma,
                              const float32_t *beta, const float32_t *stat,
                              float32_t *Y,
                              int32_t N, int32_t C, int32_t H, int32_t W,
                              int32_t num_groups,
                              int32_t start_group, int32_t end_group);

void GroupNormGradXStat_fp32_fp32(const float32_t *dY,
                                  const float32_t *X,
                                 const float32_t *gamma,
                                  const float32_t *stat,
                                  float32_t *grad_stat,
                                  int32_t N, int32_t C, int32_t H, int32_t W,
                                int32_t num_groups);

/**
 * @brief Compute the gradient of GroupNorm with respect to beta (bias parameter).
 *
 * dBeta[c] = sum over (N, H, W) of dY[n, c, h, w]
 *
 * Supports HW tiling - accumulates partial sums across tiles.
 *
 * @param dY        Upstream gradient [N, C, H, W] (tiled)
 * @param dBeta     Output gradient for beta [C] (accumulated)
 * @param N         Batch size
 * @param C         Number of channels
 * @param H         Height of this tile
 * @param W         Width of this tile
 */
void GroupNormGradB_fp32_fp32(const float32_t *dY, float32_t *dBeta,
                               int32_t N, int32_t C, int32_t H, int32_t W);

#endif // __DEEPLOY_MATH_GROUPNORM_KERNEL_HEADER_
