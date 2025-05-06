/* =====================================================================
 * Title:        DWConvolution_s8.c
 * Description:
 *
 * Date:         05.01.2023
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

#include "DeeployBasicMath.h"

void DWConv2d_s8_s8_s32_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                             uint32_t H, uint32_t W,
                             int8_t const *__restrict__ pSrcB, uint32_t P,
                             uint32_t Q, uint32_t SP, uint32_t SQ,
                             int32_t *__restrict__ pDstC, int32_t input_offset,
                             int32_t output_offset) {

  // WIESEP: For now assume padding=0
  uint32_t H_out = (H - P) / SP + 1;
  uint32_t W_out = (W - Q) / SQ + 1;

  uint32_t c = 0; // input channel loop counter
  uint32_t h = 0; // input row loop counter
  uint32_t w = 0; // input column loop counter

  uint32_t f = 0; // kernel filter loop counter
  uint32_t p = 0; // kernel row loop counter
  uint32_t q = 0; // kernel column loop counter

  int32_t sum;
  for (c = 0; c < C; ++c) {
    for (h = 0; h < H_out; ++h) {
      for (w = 0; w < W_out; ++w) {
        sum = 0;
        // printf("(%2ld,%2ld,%2ld) ", c, h, w);
        for (p = 0; p < P; ++p) {
          for (q = 0; q < Q; ++q) {
            sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
                    input_offset) *
                   pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
            // printf("%4d*%-4d + ", pSrcA[c * H * W + (h * SP + p) * W + (w *
            // SQ + q)], pSrcB[f * C * P * Q + c * P * Q + p * Q + q]);
          }
        }
        // printf("\r\n");
        // printf("= %-6ld\r\n", sum);
        pDstC[c * H_out * W_out + h * W_out + w] = sum + output_offset;
      }
    }
  }
}
