/*
 * SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULP_MaxPool2d_fp32_fp32_HWC(const float32_t *__restrict__ pSrcA,
                                  uint32_t H, uint32_t W, uint32_t C,
                                  uint32_t kernel_h, uint32_t kernel_w,
                                  uint32_t stride_h, uint32_t stride_w,
                                  float32_t *__restrict__ pDstC,
                                  uint32_t pad_top, uint32_t pad_bottom,
                                  uint32_t pad_left, uint32_t pad_right) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C);

  uint32_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
  uint32_t W_out = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

  for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
      for (uint32_t c = ch_start; c < ch_stop; ++c) {
        float32_t max_val = -inf;

        int32_t h_in_start = h_out * stride_h - pad_top;
        int32_t w_in_start = w_out * stride_w - pad_left;
        for (uint32_t kh = 0; kh < kernel_h; ++kh) {
          int32_t h_in = h_in_start + kh;

          if (h_in < 0 || h_in >= (int32_t)H) {
            continue;
          }

          for (uint32_t kw = 0; kw < kernel_w; ++kw) {
            int32_t w_in = w_in_start + kw;

            if (w_in < 0 || w_in >= (int32_t)W) {
              continue;
            }

            uint32_t input_idx = (h_in * W + w_in) * C + c;
            float32_t val = pSrcA[input_idx];

            if (val > max_val) {
              max_val = val;
            }
          }
        }

        uint32_t output_idx = (h_out * W_out + w_out) * C + c;
        pDstC[output_idx] = max_val;
      }
    }
  }
}
