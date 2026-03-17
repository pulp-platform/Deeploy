/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER_
#define __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER_

#include "DeeployPULPMath.h"

/**
 * @brief Forward LayerNorm: y = (x - mean) / inv_std_dev * scale + bias
 *
 * Parallelized across sequence positions (axis=-1 normalization).
 * Writes mean and inv_std_dev stash tensors for use by the backward pass.
 *
 * @param data_in        Input tensor [seq_length, lastDimLength]
 * @param data_out       Output tensor [seq_length, lastDimLength]
 * @param scale          Gain (gamma) [lastDimLength]
 * @param bias           Bias (beta)  [lastDimLength]
 * @param mean_out       Stash: per-sequence mean [seq_length]
 * @param inv_std_dev_out Stash: per-sequence 1/sqrt(var+eps) [seq_length]
 * @param size           Total number of elements (seq_length * lastDimLength)
 * @param lastDimLength  Normalization dimension size
 * @param epsilon        Numerical stability constant
 */
void PULP_Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias,
                              uint32_t size, uint32_t lastDimLength,
                              float32_t epsilon);

/**
 * @brief Backward LayerNorm: compute dX for a chunk of sequences.
 *
 * Uses pre-computed mean and inv_std_dev stash from the forward pass.
 * Parallelized: each core calls this for its own chunk of sequences.
 *
 * @param dy          Upstream gradient chunk [elem_count]
 * @param x           Forward input chunk [elem_count]
 * @param mean        Stash mean for this chunk [chunk_seq_count]
 * @param inv_std_dev Stash inv_std_dev for this chunk [chunk_seq_count]
 * @param dx          Output: input gradient chunk [elem_count]
 * @param gamma       Scale parameter [lastDimLength]
 * @param elem_count  Number of elements in chunk
 * @param lastDimLength Feature dimension size
 */
void PULP_LayernormGrad_fp32_fp32(const float32_t *dy, const float32_t *x,
                             const float32_t *mean, const float32_t *inv_std_dev,
                             float32_t *dx, const float32_t *gamma,
                             uint32_t elem_count, uint32_t lastDimLength);

/**
 * @brief Backward LayerNorm: compute dscale and dbias over all sequences.
 *
 * Uses pre-computed mean and inv_std_dev stash from the forward pass.
 * Single-core (core 0) operation.
 *
 * @param dy          Full upstream gradient [size]
 * @param x           Full forward input [size]
 * @param mean        Full stash mean [seq_length]
 * @param inv_std_dev Full stash inv_std_dev [seq_length]
 * @param dgamma      Output: scale gradient [lastDimLength]
 * @param dbeta       Output: bias gradient [lastDimLength]
 * @param size        Total number of elements (seq_length * lastDimLength)
 * @param lastDimLength Feature dimension size
 */
void PULP_LayernormGradParam_fp32_fp32(const float32_t *dy, const float32_t *x,
                                  const float32_t *mean, const float32_t *inv_std_dev,
                                  float32_t *dgamma, float32_t *dbeta,
                                  uint32_t size, uint32_t lastDimLength);

#endif // __DEEPLOY_MATH_LAYERNORM_KERNEL_HEADER__
