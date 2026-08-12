/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_ADD_KERNEL_HEADER_
#define __DEEPLOY_MATH_ADD_KERNEL_HEADER_

#include "DeeploySnitchMath.h"

void Add_fp32(float32_t *input1, float32_t *input2, float32_t *output,
              uint32_t size, uint32_t is_scalar);

void SnitchAdd(int8_t *pIn1, int8_t *pIn2, int32_t *pOut, uint32_t size,
               int32_t offset);

void snitch_nn_add_i8_i8_i8(
    int8_t *pIn1, int8_t *pIn2, int8_t *pOut, int32_t in1_mul, int32_t in1_add,
    uint16_t in1_shift, int32_t in2_mul, int32_t in2_add, uint16_t in2_shift,
    int32_t out_mul, int32_t out_add, uint16_t out_shift, uint16_t dim_im_in_x,
    uint16_t dim_im_in_y, uint16_t ch_im_in, int out_requant_flag);

#endif // __DEEPLOY_MATH_ADD_KERNEL_HEADER_
