// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeployBasicMath.h"

void ConvTranspose2d_fp32(const float32_t *input, uint32_t C_in, uint32_t H_in,
                          uint32_t W_in, const float32_t *weight,
                          uint32_t C_out, uint32_t groups, uint32_t K_h,
                          uint32_t K_w, uint32_t stride_h, uint32_t stride_w,
                          uint32_t dilation_h, uint32_t dilation_w,
                          uint32_t pad_top, uint32_t pad_left,
                          const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t H_out, uint32_t W_out) {
  /*
  input:       [C_in, H_in, W_in]
  weight:      [C_in, C_out / groups, K_h, K_w]
  output:      [C_out, H_out, W_out]
  bias:        [C_out] optionally
  H_out/W_out include bottom/right padding and output_padding from shape
  inference.
  */

  for (uint32_t c = 0; c < C_out; ++c) {
    for (uint32_t h = 0; h < H_out; ++h) {
      for (uint32_t w = 0; w < W_out; ++w) {
        output[(c * H_out + h) * W_out + w] = 0.0f;
      }
    }
  }

  uint32_t ch_in_per_group = C_in / groups;
  uint32_t ch_out_per_group = C_out / groups;
  for (uint32_t cout = 0; cout < C_out; ++cout) {
    uint32_t group_idx = cout / ch_out_per_group;
    uint32_t cout_in_group = cout % ch_out_per_group;
    uint32_t cin_start = group_idx * ch_in_per_group;
    for (uint32_t cin = cin_start; cin < cin_start + ch_in_per_group; ++cin) {
      for (uint32_t h_in = 0; h_in < H_in; ++h_in) {
        for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
          float32_t val = input[(cin * H_in + h_in) * W_in + w_in];
          for (uint32_t kh = 0; kh < K_h; ++kh) {
            int32_t h_out =
                (int32_t)(h_in * stride_h + kh * dilation_h) - (int32_t)pad_top;
            if (h_out < 0 || h_out >= (int32_t)H_out) {
              continue;
            }
            for (uint32_t kw = 0; kw < K_w; ++kw) {
              int32_t w_out = (int32_t)(w_in * stride_w + kw * dilation_w) -
                              (int32_t)pad_left;
              if (w_out < 0 || w_out >= (int32_t)W_out) {
                continue;
              }
              float32_t wgt =
                  weight[((cin * ch_out_per_group + cout_in_group) * K_h + kh) *
                             K_w +
                         kw];
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
