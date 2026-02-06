/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_HARDSWISH_KERNEL_HEADER_
#define __DEEPLOY_MATH_HARDSWISH_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

/*
 * HardSwish Activation Function
 *
 * Computes: HardSwish(x) = x * clip(x/6 + 0.5, 0, 1)
 *
 * Piecewise form:
 *   - When x <= -3: output = 0
 *   - When -3 < x < 3: output = x * (x/6 + 0.5)
 *   - When x >= 3: output = x
 *
 * This is a computationally efficient approximation of Swish/SiLU activation
 * commonly used in mobile neural networks and transformer models.
 *
 * data_in:  Input tensor (FP32)
 * data_out: Output tensor (FP32, same shape as input)
 * size:     Total number of elements
 *
 * multi-core      = yes
 * parallelization = element-wise
 */
void HardSwish_fp32(float32_t *data_in, float32_t *data_out, uint32_t size);

#endif // __DEEPLOY_MATH_HARDSWISH_KERNEL_HEADER_
