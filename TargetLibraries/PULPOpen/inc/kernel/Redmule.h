
/* =====================================================================
 * Title:        Matmul_redmule.h
 * Description:
 *
 * $Date:        05.06.2025
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Run Wang, ETH Zurich
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

#include "DeeployPULPMath.h"


void MatMul_fp32_fp32_fp32_Redmule(
    const float32_t *__restrict__ pSrcA,
    const float32_t *__restrict__ pSrcB,
    float32_t *__restrict__ pDstY,
    uint32_t M,
    uint32_t N,
    uint32_t O);


void Gemm_fp32_fp32_fp32_fp32_Redmule(
    const float32_t *__restrict__ pSrcA,
    const float32_t *__restrict__ pSrcB,
    const float32_t *__restrict__ pBias,
    float32_t *__restrict__ pDstY,
    uint32_t M,
    uint32_t N,
    uint32_t O);

void Conv2d_Im2Col_fp32_fp32_fp32_HWC_8_Redmule(
    const float32_t *__restrict__ pSrcA,
    uint32_t H,
    uint32_t W,
    uint32_t C,
    const float32_t *__restrict__ pSrcB,
    uint32_t P,
    uint32_t Q,
    uint32_t SP,
    uint32_t SQ,
    float32_t *__restrict__ pDstC,
    uint32_t F,
    uint32_t pad_top,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t pad_right,
    float32_t *__restrict__ pIm2ColBuffer);