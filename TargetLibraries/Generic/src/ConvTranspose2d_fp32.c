// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeployBasicMath.h"

void ConvTranspose2d_fp32(const float32_t *input, uint32_t C_in, uint32_t H_in,
                          uint32_t W_in, const float32_t *weight,
                          uint32_t C_out, uint32_t K_h, uint32_t K_w,
                          uint32_t stride_h, uint32_t stride_w,
                          const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t H_out, uint32_t W_out) {
  /*
  input:       [C_in, H_in, W_in]
  weight:      [C_in, C_out, K_h, K_w]
  output:      [C_out, H_out, W_out]
  bias:        [C_out] optionally
  */

  for (uint32_t c = 0; c < C_out; ++c) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        output[(c * H_out + h) * W_out + w] = 0.0f;
      }
    }
  }

  for (uint32_t cout = 0; cout < C_out; ++cout) {
    for (uint32_t cin = 0; cin < C_in; ++cin) {
      for (uint32_t h_in = 0; h_in < H_in; ++h_in) {
        for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
          float32_t val = input[(cin * H_in + h_in) * W_in + w_in];
          for (uint32_t kh = 0; kh < K_h; ++kh) {
            uint32_t h_out = h_in * stride_h + kh;
            if (h_out >= H_out) {
              continue;
            }
            for (uint32_t kw = 0; kw < K_w; ++kw) {
              uint32_t w_out = w_in * stride_w + kw;
              if (w_out >= W_out) {
                continue;
              }
              float32_t wgt =
                  weight[((cin * C_out + cout) * K_h + kh) * K_w + kw];
              output[(cout * H_out + h_out) * W_out + w_out] += val * wgt;
            }
          }
        }
      }
    }

    if (has_bias) {
      for (uint32_t h = 0; h < H_out; ++h) {
        for (uint32_t w = 0; w < W_out; ++w) {
          output[(cout * H_out + h) * W_out + w] += bias[cout];
        }
      }
    }
  }
}
