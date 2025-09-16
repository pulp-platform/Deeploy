/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_RQGELU_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RQGELU_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This file implements the requantized GELU.
 */

/******************************************************************************/
/*                 GELU with requantization to 8bit                       */
/******************************************************************************/

void RQGELU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize, int8_t b,
                  int16_t one, int32_t input_offset, int32_t output_offset,
                  int32_t *mul, int32_t *add, int32_t *shift);

#endif //__DEEPLOY_BASIC_MATH_RQGELU_KERNEL_HEADER_
