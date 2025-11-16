/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_SOFTMAX_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SOFTMAX_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/*
 * This file implements various softmax kernels.
 */

/******************************************************************************/
/*                              Softmax (8bit)                                */
/******************************************************************************/

/**
 * @brief Approximate softmax implementation according to the I-BERT paper.
 * @see https://arxiv.org/abs/2101.01321
 *
 * @param data_in
 * @param data_out
 * @param size
 * @param lastDimLength
 * @param coeffA
 * @param coeffB
 * @param coeffC
 * @param log2
 * @param n_levels
 */
void Softmax_s8_s8(int8_t *data_in, int8_t *data_out, uint32_t size,
                   uint32_t lastDimLength, int32_t coeffA, int32_t coeffB,
                   int64_t coeffC, int32_t log2, uint32_t n_levels);

/**
 * @brief Approximate softmax implementation.
 *
 * @param pSrcA
 * @param pDstB
 * @param pBufN
 * @param size
 * @param lastDimLength
 * @param n_levels
 */
void ITAMax_s8(int8_t const *__restrict__ pSrcA, int8_t *__restrict__ pDstB,
               int8_t *__restrict__ pBufN, uint32_t size,
               uint32_t lastDimLength, uint32_t n_levels);

/**
 * @brief Approximate partial softmax implementation used in ITA.
 *
 * @param pSrcA
 * @param pDstB
 * @param size
 * @param lastDimLength
 * @param group_width
 * @param n_levels
 */
void ITAPartialMax_s8(int8_t const *__restrict__ pSrcA,
                      int8_t *__restrict__ pDstB, uint32_t size,
                      uint32_t lastDimLength, uint32_t group_width,
                      uint32_t n_levels);

void Softmax_fp32_fp32(float32_t *input, float32_t *output, int32_t size,
                       int32_t last_dim_length);

void SoftmaxGrad_fp32_fp32_fp32(float32_t *upstream_grad,
                                float32_t *softmax_output,
                                float32_t *softmax_gradient, int32_t size,
                                int32_t last_dim_length);
#endif //__DEEPLOY_BASIC_MATH_SOFTMAX_KERNEL_HEADER_
