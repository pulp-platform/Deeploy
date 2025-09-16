/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_RQDIV_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RQDIV_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This file implements the requantized division.
 */

/******************************************************************************/
/*                 Division with requantization to 8bit                       */
/******************************************************************************/

void RQDiv_s32_s8(int32_t *data_in_nom, int32_t *data_in_denom,
                  int32_t size_nom, int32_t size_denom, int32_t nomStep,
                  int32_t denomStep, int8_t *data_out, int32_t Delta,
                  int32_t eps, int32_t eta, int32_t requant_mul,
                  int32_t requant_add, int32_t requant_shift);

#endif //__DEEPLOY_BASIC_MATH_RQDIV_KERNEL_HEADER_
