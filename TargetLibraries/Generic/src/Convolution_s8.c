/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"

void Conv2d_s8_s8_s32_NCHW(int8_t const *__restrict__ pSrcA, uint32_t C,
                           uint32_t H, uint32_t W,
                           int8_t const *__restrict__ pSrcB, uint32_t F,
                           uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
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
  for (f = 0; f < F; ++f) {
    for (h = 0; h < H_out; ++h) {
      for (w = 0; w < W_out; ++w) {
        sum = 0;
        for (c = 0; c < C; ++c) {
          // printf("(%2d,%2d,%2d) ", c, h, w);
          for (p = 0; p < P; ++p) {
            for (q = 0; q < Q; ++q) {
              sum += (pSrcA[c * H * W + (h * SP + p) * W + (w * SQ + q)] +
                      input_offset) *
                     pSrcB[f * C * P * Q + c * P * Q + p * Q + q];
              // printf("%4d*%-4d + ", pSrcA[c * H * W + (h * SP + p) * W + (w *
              // SQ + q)],
              //  pSrcB[f * C * P * Q + c * P * Q + p * Q + q]);
            }
          }
          // printf("\r\n");
        }
        // printf("= %-6ld\r\n", sum);
        pDstC[f * H_out * W_out + h * W_out + w] = sum + output_offset;
      }
    }
  }
}
