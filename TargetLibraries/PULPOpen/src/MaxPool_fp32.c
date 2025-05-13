/* =====================================================================
 * Title:        MaxPool_fp32.c
 * Description:
 *
 * Date:         27.01.2025
 *
 * ===================================================================== */

/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
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
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void MaxPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA, uint32_t H,
                             uint32_t W, uint32_t C, uint32_t P, uint32_t Q,
                             uint32_t SP, uint32_t SQ,
                             float32_t *__restrict__ pDstC, uint32_t pad_top,
                             uint32_t pad_bottom, uint32_t pad_left,
                             uint32_t pad_right) {

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t h = 0; h < H_out; ++h) {
    for (uint32_t w = 0; w < W_out; ++w) {

      for (uint32_t c = 0; c < C; ++c) {
        float32_t max_val = -inf;

        for (uint32_t p = 0; p < P; ++p) {
          for (uint32_t q = 0; q < Q; ++q) {
            int32_t h_in = h * SP + p - pad_top;
            int32_t w_in = w * SQ + q - pad_left;

            if (h_in < 0 || h_in >= H || w_in < 0 || w_in >= W) {
              continue;
            }

            uint32_t input_idx = (h_in * W + w_in) * C + c;
            float32_t val = pSrcA[input_idx];

            if (val > max_val) {
              max_val = val;
            }
          }
        }

        uint32_t output_idx = (h * W_out + w) * C + c;

        pDstC[output_idx] = max_val;
      }
    }
  }
}