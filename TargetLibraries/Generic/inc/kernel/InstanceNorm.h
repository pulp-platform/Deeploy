/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_INSTANCENORM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_INSTANCENORM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                         Instance Normalization                             */
/******************************************************************************/
void InstanceNormalization_fp32_fp32(const float32_t *__restrict__ src,
                                     float32_t *__restrict__ dst,
                                     const float32_t *__restrict__ scale,
                                     const float32_t *__restrict__ bias,
                                     uint32_t batch_size, uint32_t num_channels,
                                     uint32_t spatial, float32_t epsilon);

#endif //__DEEPLOY_BASIC_MATH_INSTANCENORM_KERNEL_HEADER_
