/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_GEMV_KERNEL_HEADER_
#define __DEEPLOY_MATH_GEMV_KERNEL_HEADER_

#include "stdint.h"
#include "DeeployPULPMath.h"

void gemv_s8_s8_plp(int8_t *pIn, int8_t *pBias, int8_t *pOut, int8_t *pWeight,
                    int32_t *pKappa, int32_t *pLambda, uint16_t out_mult,
                    uint16_t out_shift, uint16_t dim_vec,
                    uint16_t num_o_neurons, uint8_t flag_relu,
                    uint8_t flag_batch_norm);

#endif // __DEEPLOY_MATH_GEMV_KERNEL_HEADER_
