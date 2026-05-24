/* =====================================================================
 * Title:        iLeakyReLU.h  (SoCDAML Part III - Step 3, provided)
 *
 * Header for the iLeakyReLU PULP kernel.
 * Drop into: TargetLibraries/PULPOpen/inc/kernel/iLeakyReLU.h
 * and add `#include "kernel/iLeakyReLU.h"` to DeeployPULPMath.h.
 * ===================================================================== */
/* SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_KERNEL_ILEAKYRELU_H_
#define __DEEPLOY_KERNEL_ILEAKYRELU_H_

#include "DeeployPULPMath.h"

void PULPiLeakyReLU_i8_i8(int8_t *pIn, int8_t *pOut, uint32_t size, int32_t mul,
                          int32_t shift);

#endif // __DEEPLOY_KERNEL_ILEAKYRELU_H_
