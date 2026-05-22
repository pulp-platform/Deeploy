/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void AveragePool2d_fp32_fp32(float32_t const *__restrict__ src,
                             float32_t *__restrict__ dst, uint32_t N,
                             uint32_t C, uint32_t H, uint32_t W,
                             uint32_t kernel_h, uint32_t kernel_w,
                             uint32_t stride_h, uint32_t stride_w,
                             uint32_t pad_top, uint32_t pad_left,
                             uint32_t pad_bottom, uint32_t pad_right) {

  if (N == 0 || C == 0 || stride_h == 0 || stride_w == 0 ||
      (H + pad_top + pad_bottom) < kernel_h ||
      (W + pad_left + pad_right) < kernel_w) {
    return;
  }

  uint32_t H_out = (H + pad_top + pad_bottom - kernel_h) / stride_h + 1;
  uint32_t W_out = (W + pad_left + pad_right - kernel_w) / stride_w + 1;

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t c = 0; c < C; ++c) {
      for (uint32_t h_out = 0; h_out < H_out; h_out++) {
        for (uint32_t w_out = 0; w_out < W_out; w_out++) {

          float32_t sum = 0.0f;
          uint32_t count = 0;

          for (uint32_t kh = 0; kh < kernel_h; kh++) {
            for (uint32_t kw = 0; kw < kernel_w; kw++) {

              int32_t h_in = (int32_t)(h_out * stride_h + kh) - pad_top;
              int32_t w_in = (int32_t)(w_out * stride_w + kw) - pad_left;

              if (h_in >= 0 && h_in < (int32_t)H && w_in >= 0 &&
                  w_in < (int32_t)W) {
                sum += src[((n * C + c) * H + h_in) * W + w_in];
                count++;
              }
            }
          }
          uint32_t idx = ((n * C + c) * H_out + h_out) * W_out + w_out;
          dst[idx] = sum / (float32_t)count;
        }
      }
    }
  }
}

void AveragePool1d_fp32_fp32(float32_t const *__restrict__ src,
                             float32_t *__restrict__ dst, uint32_t N,
                             uint32_t C, uint32_t L, uint32_t kernel_len,
                             uint32_t stride, uint32_t pad_left,
                             uint32_t pad_right) {

  if (N == 0 || C == 0 || stride == 0 ||
      (L + pad_left + pad_right) < kernel_len) {
    return;
  }

  uint32_t L_out = (L + pad_left + pad_right - kernel_len) / stride + 1;

  for (uint32_t n = 0; n < N; ++n) {
    for (uint32_t c = 0; c < C; ++c) {
      for (uint32_t l_out = 0; l_out < L_out; l_out++) {

        float32_t sum = 0.0f;
        uint32_t count = 0;

        for (uint32_t k = 0; k < kernel_len; k++) {

          int32_t l_in = (int32_t)(l_out * stride + k) - (int32_t)pad_left;

          if (l_in >= 0 && l_in < (int32_t)L) {
            sum += src[(n * C + c) * L + l_in];
            count++;
          }
        }
        uint32_t i = (n * C + c) * L_out + l_out;
        dst[i] = (count == 0) ? 0.0f : (sum / (float32_t)count);
      }
    }
  }
}