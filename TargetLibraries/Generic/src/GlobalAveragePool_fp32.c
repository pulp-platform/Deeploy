/* =====================================================================
 * Title:        GlobalAveragePool_fp32.c
 * Description:  Float32 version of GlobalAveragePool with NCHW format
 *
 * Date:         18.08.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2025 <Your Name>, ETH Zurich and University of Bologna.
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
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployBasicMath.h"

void GlobalAveragePool_fp32_NCHW(const float32_t *__restrict__ pSrc, uint32_t N,
                                uint32_t C, uint32_t H, uint32_t W,
                                float32_t *__restrict__ pDst) {
    for (uint32_t n = 0; n < N; n++) {
        for (uint32_t c = 0; c < C; c++) {
            float32_t sum = 0.0f;
            for (uint32_t h = 0; h < H; h++) {
                for (uint32_t w = 0; w < W; w++) {
                    sum += pSrc[n * C * H * W + c * H * W + h * W + w];
                }
            }
            pDst[n * C + c] = sum / (float32_t)(H * W);
        }
    }
}