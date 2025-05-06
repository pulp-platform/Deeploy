/* =====================================================================
 * Title:        RMSNorm.h
 * Description:
 *
 * $Date:        20.02.2024
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef __DEEPLOY_BASIC_MATH_RMSNORM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RMSNORM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 *
 */

/******************************************************************************/
/*                             Layernorm (8bit)                               */
/******************************************************************************/

void iRMSnorm_s8_s8(int8_t *data_in, int8_t *data_out, int32_t *weight,
                    int32_t input_offset, int32_t size, int32_t lastDimLength,
                    int32_t log2D);

#endif //__DEEPLOY_BASIC_MATH_RMSNORM_KERNEL_HEADER_
