/* =====================================================================
 * Title:        Convolution_float32.c
 * Description:  Float32 version of Conv2D with NCHW format (pre-padded input)
 *
 * Date:         23.01.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Run Wang, ETH Zurich
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

#include "DeeployPULPMath.h"
#include "pmsis.h"

void Conv2d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float32_t *__restrict__ pDstC,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {



    uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
    uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;


    uint32_t h, w, c, f, p, q;

    for (h = 0; h < H_out; ++h) {
        for (w = 0; w < W_out; ++w) {
            
            for (f = 0; f < F; ++f) {
                float32_t sum = 0.0f;
                
                for (p = 0; p < P; ++p) {
                    for (q = 0; q < Q; ++q) {
                        for (c = 0; c < C; ++c) {
                            int32_t h_in = h * SP + p - pad_top;
                            int32_t w_in = w * SQ + q - pad_left;

                            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) {
                                continue;
                            }

                            uint32_t input_idx = (h_in * W + w_in) * C + c;
                            uint32_t weight_idx = f * (P * Q * C) + p * (Q * C) + q * C + c;

                            float32_t input_val = pSrcA[input_idx];
                            float32_t weight_val = pSrcB[weight_idx];
                            float32_t mult_result = input_val * weight_val;

                            sum += mult_result;
                        }
                    }
                }

                uint32_t output_idx = (h * W_out + w) * F + f;
                pDstC[output_idx] = sum;
            }
        }
    }
}