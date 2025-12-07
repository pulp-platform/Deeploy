/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Generic_AvgPool2d_fp32_fp32_NCHW(const float *__restrict__ pSrcA,
                                      uint32_t W, uint32_t H, uint32_t C,
                                      uint32_t Q, uint32_t P, uint32_t SQ,
                                      uint32_t SP, float *__restrict__ pDstC,
                                      uint32_t pad_top, uint32_t pad_bottom,
                                      uint32_t pad_left, uint32_t pad_right) {

  // NCHW layout: N x C x H x W
  // W = width (dim_im_in_x)
  // H = height (dim_im_in_y)
  // Input access: input[c][h][w] = input[c * H * W + h * W + w]

  uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
  uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;

  for (uint32_t c = 0; c < C; ++c) {
    for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
      for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
        float sum = 0.0f;
        uint32_t count = 0;

        int32_t h_in_start = (int32_t)h_out * (int32_t)SP - (int32_t)pad_top;
        int32_t w_in_start = (int32_t)w_out * (int32_t)SQ - (int32_t)pad_left;

        for (uint32_t p = 0; p < P; ++p) {
          int32_t h_in = h_in_start + (int32_t)p;
          if (h_in < 0 || h_in >= (int32_t)H) continue;

          for (uint32_t q = 0; q < Q; ++q) {
            int32_t w_in = w_in_start + (int32_t)q;
            if (w_in < 0 || w_in >= (int32_t)W) continue;

            // NCHW layout: input[c][h][w]
            uint32_t input_idx = c * H * W + (uint32_t)h_in * W + (uint32_t)w_in;
            sum += pSrcA[input_idx];
            count++;
          }
        }

        // NCHW layout: output[c][h_out][w_out]
        uint32_t output_idx = c * H_out * W_out + h_out * W_out + w_out;
        pDstC[output_idx] = (count > 0) ? (sum / (float)count) : 0.0f;
      }
    }
  }
}
