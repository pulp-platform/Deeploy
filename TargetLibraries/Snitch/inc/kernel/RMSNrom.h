/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_RMSNORM_KERNEL_HEADER_
#define __DEEPLOY_MATH_RMSNORM_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

/*
 * RMS Normalization (Root Mean Square Normalization)
 *
 * Computes: output[i] = (input[i] / rms) * weight[i]
 * where rms = sqrt(mean(input^2) + eps)
 *
 * data_in:        Input tensor [batch, seq, hidden] or flattened [size]
 * weight:         Weight tensor [hidden_dim]
 * data_out:       Output tensor (same shape as input)
 * size:           Total number of elements (batch * seq * hidden)
 * lastDimLength:  Hidden dimension size
 * eps:            Epsilon for numerical stability (typically 1e-6)
 *
 * multi-core      = yes
 * parallelization = vector-wise (across batch * sequence)
 */
void RMSNorm_fp32(float32_t *data_in, float32_t *weight, float32_t *data_out,
                  uint32_t size, uint32_t lastDimLength, float32_t eps);

#endif // __DEEPLOY_MATH_RMSNORM_KERNEL_HEADER_
