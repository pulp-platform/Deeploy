/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

// Convolution on already padded input, focus on an output window, make the
// addition flowing onto a PxQ window on input along with the weight window,
// changing channel and repeat the procedure. Then change the output window and
// repeat the procedure. Then change the filter and repeat.

#include "Conv.h"

void Carfield_Conv(float32_t *input, float32_t *weight, float32_t *bias,
                   float32_t *output, int32_t F, int32_t C, int32_t H_in,
                   int32_t W_in, int32_t P, int32_t Q, int32_t core_id,
                   int32_t numThreads) {

  int32_t H_out = H_in - P + 1;
  int32_t W_out = W_in - Q + 1;

  for (int f = core_id; f < F;
       f += numThreads) { // parallelization over filters, every core takes a
                          // bunch of filters.
    for (int h = 0; h < H_out; h++) {
      for (int w = 0; w < W_out; w++) {
        float32_t acc = bias[f];
        for (int c = 0; c < C; c++) {
          for (int p = 0; p < P; p++) {
            for (int q = 0; q < Q; q++) {
              // input[c][h+p][w+q] * weight[f][c][p][q], h+p=h_in no stride,
              // w+q=w_in c=channel, h=output height, w=output width, p=kernel
              // height, q=kernel width
              acc += input[c * (H_in * W_in) + (h + p) * W_in + (w + q)] *
                     weight[f * (C * P * Q) + c * (P * Q) + p * (Q) + q];
            }
          }
        }
        output[f * (H_out * W_out) + h * (W_out) + w] = acc;
      }
    }
  }
}
