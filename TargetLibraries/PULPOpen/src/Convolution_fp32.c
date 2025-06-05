/* =====================================================================
 * Title:        Conv.c
 * Description:  Float32 version of Conv2D with NCHW format (pre-padded input)
 *
 * Date:         05.06.2025
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

void PULP_Conv2d_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA, uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB, uint32_t F_total,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float32_t *__restrict__ pDstC, 
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {

    int8_t core_id = pi_core_id();
    int8_t log2Core = log2(NUM_CORES);
    
    uint16_t ch_out_chunk = (F_total >> log2Core) + ((F_total & (NUM_CORES-1)) != 0);
    uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
    uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
    uint16_t ch_out_count = ch_out_stop - ch_out_start;

    if (ch_out_count == 0) {
        return;
    }

    const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;

    uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
    uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

    for (uint32_t h = 0; h < H_out; ++h) {
        for (uint32_t w = 0; w < W_out; ++w) {
            for (uint32_t f = 0; f < ch_out_count; ++f) {
                float32_t sum = 0.0f;
                
                for (uint32_t p = 0; p < P; ++p) {
                    for (uint32_t q = 0; q < Q; ++q) {
                        for (uint32_t c = 0; c < C; ++c) {
                            int32_t h_in = h * SP + p - pad_top;
                            int32_t w_in = w * SQ + q - pad_left;

                            if (h_in < 0 || h_in >= (int32_t)H || w_in < 0 || w_in >= (int32_t)W) {
                                continue;
                            }

                            uint32_t input_idx = (h_in * W + w_in) * C + c;
                            uint32_t weight_idx = f * (P * Q * C) + p * (Q * C) + q * C + c;

                            sum += pSrcA[input_idx] * weight_ptr[weight_idx];
                        }
                    }
                }

                uint32_t output_idx = (h * W_out + w) * F_total + (ch_out_start + f);
                pDstC[output_idx] = sum;
            }
        }
    }
}

void PULP_Conv2d_Im2Col_fp32_fp32_fp32_HWC(
    const float32_t *__restrict__ pSrcA,
    uint32_t H, uint32_t W, uint32_t C,
    const float32_t *__restrict__ pSrcB,
    uint32_t F_total,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float32_t *__restrict__ pDstC,
    uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float32_t *__restrict__ pContextBuffer)
{
    int8_t core_id = pi_core_id();
    int8_t log2Core = log2(NUM_CORES);
    
    uint16_t ch_out_chunk = (F_total >> log2Core) + ((F_total & (NUM_CORES-1)) != 0);
    uint16_t ch_out_start = MIN(ch_out_chunk * core_id, F_total);
    uint16_t ch_out_stop = MIN(ch_out_start + ch_out_chunk, F_total);
    uint16_t ch_out_count = ch_out_stop - ch_out_start;

    if (ch_out_count == 0) {
        return;
    }

    const float32_t *weight_ptr = pSrcB + ch_out_start * C * P * Q;
    
    uint32_t im2col_size_per_core = C * P * Q;
    float32_t *im2col_buffer = pContextBuffer + core_id * im2col_size_per_core;

    uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
    uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
    uint32_t kernel_size = P * Q * C;

    for (uint32_t h_out = 0; h_out < H_out; h_out++)
    {
        for (uint32_t w_out = 0; w_out < W_out; w_out++)
        {
            int32_t h_in_start = h_out * SP - pad_top;
            int32_t w_in_start = w_out * SQ - pad_left;

            for (uint32_t p = 0; p < P; p++)
            {
                int32_t h_in = h_in_start + p;

                for (uint32_t q = 0; q < Q; q++)
                {
                    int32_t w_in = w_in_start + q;

                    for (uint32_t c = 0; c < C; c++)
                    {
                        if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 && w_in < (int32_t)W)
                        {
                            uint32_t in_idx = (h_in * W + w_in) * C + c;
                            im2col_buffer[p * Q * C + q * C + c] = pSrcA[in_idx];
                        }
                        else
                        {
                            im2col_buffer[p * Q * C + q * C + c] = 0.0f;
                        }
                    }
                }
            }

            for (uint32_t f = 0; f < ch_out_count; f++)
            {
                float32_t sum = 0.0f;
                const float32_t *local_weight_ptr = weight_ptr + f * kernel_size;

                for (uint32_t k = 0; k < kernel_size; k++)
                {
                    sum += im2col_buffer[k] * local_weight_ptr[k];
                }

                uint32_t out_idx = (h_out * W_out + w_out) * F_total + (ch_out_start + f);
                pDstC[out_idx] = sum;
            }
        }
    }
}