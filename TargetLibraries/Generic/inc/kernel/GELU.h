/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_GELU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_GELU_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 *
 */

/******************************************************************************/
/*                              Division (32bit)                              */
/******************************************************************************/

void GELU_s8_s32(int8_t *data_in, int32_t *data_out, int32_t dataSize, int8_t b,
                 int16_t one, int32_t input_offset);

void GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize);

void GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out,
                            int32_t dataSize);
                            
void GELU_fp32_fp32_sigmoid_grad_chunk(float32_t *grad_in, float32_t *data_in,
                                       float32_t *grad_out, int32_t start_idx,
                                       int32_t end_idx);

#endif //__DEEPLOY_BASIC_MATH_GELU_KERNEL_HEADER_
