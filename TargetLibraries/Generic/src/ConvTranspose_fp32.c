// SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include "DeeployBasicMath.h"

void ConvTranspose1d_fp32(const float32_t *input, uint32_t C_in, uint32_t L_in,
                          const float32_t *weight, uint32_t C_out, uint32_t K,
                          uint32_t stride, const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t L_out) {
  /*
  input:  [C_in, L_in]
  weight: [C_in, C_out, K]
  output: [C_out, L_out]
  bias:   [C_out] (optional)
  */

  for (uint32_t cout = 0; cout < C_out; ++cout) {
    float32_t b = has_bias ? bias[cout] : 0.0f;
    float32_t *out_row = output + cout * L_out;
    for (uint32_t l = 0; l < L_out; ++l)
      out_row[l] = b;
  }

  for (uint32_t cout = 0; cout < C_out; ++cout) {
    float32_t *out_row = output + cout * L_out;
    for (uint32_t cin = 0; cin < C_in; ++cin) {
      const float32_t *in_row = input + cin * L_in;
      const float32_t *wgt_row = weight + cin * (C_out * K) + cout * K;
      for (uint32_t l_in = 0; l_in < L_in; ++l_in) {
        float32_t val = in_row[l_in];
        uint32_t base = l_in * stride;
        for (uint32_t k = 0; k < K; ++k)
          out_row[base + k] += val * wgt_row[k];
      }
    }
  }
}

void ConvTranspose2d_fp32(const float32_t *input, uint32_t C_in, uint32_t H_in,
                          uint32_t W_in, const float32_t *weight,
                          uint32_t C_out, uint32_t kH, uint32_t kW,
                          uint32_t stride_h, uint32_t stride_w,
                          const float32_t *bias, bool has_bias,
                          float32_t *output, uint32_t H_out, uint32_t W_out) {
  /*
  input:  [C_in, H_in, W_in]
  weight: [C_in, C_out, kH, kW]
  output: [C_out, H_out, W_out]
  bias:   [C_out] (optional)
  */

  for (uint32_t cout = 0; cout < C_out; ++cout) {
    float32_t b = has_bias ? bias[cout] : 0.0f;
    float32_t *out_ch = output + cout * H_out * W_out;
    for (uint32_t i = 0; i < H_out * W_out; ++i)
      out_ch[i] = b;
  }

  for (uint32_t cout = 0; cout < C_out; ++cout) {
    float32_t *out_ch = output + cout * H_out * W_out;
    for (uint32_t cin = 0; cin < C_in; ++cin) {
      const float32_t *in_ch = input + cin * H_in * W_in;
      const float32_t *wgt_ch =
          weight + cin * (C_out * kH * kW) + cout * (kH * kW);
      for (uint32_t h_in = 0; h_in < H_in; ++h_in) {
        const float32_t *in_row = in_ch + h_in * W_in;
        uint32_t h_base = h_in * stride_h;
        for (uint32_t w_in = 0; w_in < W_in; ++w_in) {
          float32_t val = in_row[w_in];
          uint32_t w_base = w_in * stride_w;
          for (uint32_t kh = 0; kh < kH; ++kh) {
            float32_t *out_row = out_ch + (h_base + kh) * W_out + w_base;
            const float32_t *wgt_krow = wgt_ch + kh * kW;
            for (uint32_t kw = 0; kw < kW; ++kw)
              out_row[kw] += val * wgt_krow[kw];
          }
        }
      }
    }
  }
}
