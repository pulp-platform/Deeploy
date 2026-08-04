/* =====================================================================
 * Title:        iLeakyReLU.h
 * Description:  int8 quantization-friendly LeakyReLU.
 *               SoCDAML Part III - TA reference solution.
 *
 * out[i] = (in[i] >= 0) ? in[i] : ((mul * in[i]) >> shift)
 * ===================================================================== */
/* Copyright (C) 2026 ETH Zurich and University of Bologna.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_KERNEL_ILEAKYRELU_H_
#define __DEEPLOY_KERNEL_ILEAKYRELU_H_

#include "DeeployPULPMath.h"

void PULPiLeakyReLU_i8_i8(int8_t *pIn, int8_t *pOut, uint32_t size, int32_t mul,
                          int32_t shift);

#endif // __DEEPLOY_KERNEL_ILEAKYRELU_H_
