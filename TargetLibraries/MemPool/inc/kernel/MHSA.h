/* =====================================================================
 * Title:        MHSA.h
 * Description:
 *
 * Date:         08.02.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
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

#ifndef __DEEPLOY_MATH_MHSA_KERNEL_HEADER_
#define __DEEPLOY_MATH_MHSA_KERNEL_HEADER_

/* Includes ------------------------------------------------------------------*/

#include "DeeployMath.h"

/* This library implements multi-head self attention for several data widths
 * in multiple different ways. The functions all follow the following format:
 */

/******************************************************************************/
/*                      Multi-Head Self Attention (8bit)                      */
/******************************************************************************/

/*
 * MHSA  ----------------------------------
 * kernel      = M1HSA_s8_ITA
 * data type   = 8-bit integer
 * multi-core  = yes
 * unrolling   = no
 * simd        = no
 * accelerator = ITA
 * heads       = 1
 */
void M1HSA_s8_ITA(int8_t const *__restrict__ pSrcQ,
                  int8_t const *__restrict__ pSrcK, int8_t *__restrict__ pBuf,
                  uint32_t S, uint32_t E, uint32_t P,
                  ita_quant_t const *__restrict__ quant_param,
                  int8_t *__restrict__ pDst, int8_t Q_offset, int8_t K_offset,
                  int8_t output_offset, uint32_t core_id, uint32_t numThreads);

/*
 * MHSA  ----------------------------------
 * kernel      = M2HSA_s8_ITA
 * data type   = 8-bit integer
 * multi-core  = yes
 * unrolling   = no
 * simd        = no
 * accelerator = ITA
 * heads       = 2
 */
void M2HSA_s8_ITA(int8_t const *__restrict__ pSrcQ,
                  int8_t const *__restrict__ pSrcK, int8_t **__restrict__ pBuf,
                  uint32_t S, uint32_t E, uint32_t P,
                  ita_quant_t const **__restrict__ quant_params,
                  int8_t *__restrict__ pDst, int8_t Q_offset, int8_t K_offset,
                  int8_t output_offset, uint32_t core_id, uint32_t numThreads);

/*
 * MHSA  ----------------------------------
 * kernel      = M4HSA_s8_ITA
 * data type   = 8-bit integer
 * multi-core  = yes
 * unrolling   = no
 * simd        = no
 * accelerator = ITA
 * heads       = 4
 */
void M4HSA_s8_ITA(int8_t const *__restrict__ pSrcQ,
                  int8_t const *__restrict__ pSrcK, int8_t **__restrict__ pBuf,
                  uint32_t S, uint32_t E, uint32_t P,
                  ita_quant_t const **__restrict__ quant_params,
                  int8_t *__restrict__ pDst, int8_t Q_offset, int8_t K_offset,
                  int8_t output_offset, uint32_t core_id, uint32_t numThreads);

#endif //__DEEPLOY_MATH_MHSA_KERNEL_HEADER_
