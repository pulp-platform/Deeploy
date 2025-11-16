/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
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
