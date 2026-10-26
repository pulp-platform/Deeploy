/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "Resize.h"

void Carfield_ResizeNearest2D(float32_t *input, float32_t *output, int32_t C,
                              int32_t H_in, int32_t W_in, int32_t H_out,
                              int32_t W_out, int32_t scale_h, int32_t scale_w,
                              int32_t core_id, int32_t numThreads) {

  int32_t plane = H_out * W_out;
  int32_t size = C * plane;

  int32_t chunk = (size + numThreads - 1) / numThreads;
  int32_t start = core_id * chunk;
  int32_t end = start + chunk;
  if (end > size)
    end = size;
  if (start > size)
    start = size;

  for (int32_t o = start; o < end; o++) {
    int32_t c = o / plane;
    int32_t r = o - c * plane;
    int32_t h = r / W_out;
    int32_t w = r - h * W_out;

    output[o] = input[c * H_in * W_in + (h / scale_h) * W_in + (w / scale_w)];
  }
}
