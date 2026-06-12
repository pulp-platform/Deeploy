/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_SPATZ_MATH_HEADER_
#define __DEEPLOY_SPATZ_MATH_HEADER_

#include <stdint.h>
#include <stdbool.h>

#include "DeeployBasicMath.h"
#include "snrt.h"

void Spatz_MatMul_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
								 const float32_t *__restrict__ pSrcB,
								 float32_t *__restrict__ pDstY, uint32_t M,
								 uint32_t N, uint32_t O);

void Spatz_Softmax_fp32_fp32(float32_t *input, float32_t *output, int32_t size,
                       int32_t last_dim_length);


void compute_topk_min_heap( uint32_t k, uint32_t n, float32_t *data_in, float32_t *heap_values, int32_t *heap_indices);

// void Spatz_MatMul_fp16_fp16_fp16(const __fp16 *__restrict__ pSrcA,
// 								 const __fp16 *__restrict__ pSrcB,
// 								 __fp16 *__restrict__ pDstY, uint32_t M,
// 								 uint32_t N, uint32_t O);
// 
// void Spatz_MatMul_fp64_fp64_fp64(const double *__restrict__ pSrcA,
// 								 const double *__restrict__ pSrcB,
// 								 double *__restrict__ pDstY, uint32_t M,
// 								 uint32_t N, uint32_t O);

#define BEGIN_SINGLE_CORE if (core_id == 0) {
#define END_SINGLE_CORE }
#define SINGLE_CORE if (core_id == 0)

#endif // __DEEPLOY_SPATZ_MATH_HEADER_
