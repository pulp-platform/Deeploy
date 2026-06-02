/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_GROUPNORM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_GROUPNORM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                          Group Normalization                               */
/******************************************************************************/
void GroupNormalization_fp32_fp32(const float32_t *__restrict__ src,
                                  float32_t *__restrict__ dst,
                                  const float32_t *__restrict__ scale,
                                  const float32_t *__restrict__ bias,
                                  uint32_t batch_size, uint32_t num_channels,
                                  uint32_t spatial, uint32_t num_groups,
                                  float32_t epsilon);

#endif //__DEEPLOY_BASIC_MATH_GROUPNORM_KERNEL_HEADER_
