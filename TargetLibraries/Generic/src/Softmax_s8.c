/* =====================================================================
 * Title:        Softmax_s8.c
 * Description:
 *
 * $Date:        27.03.2023
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * - Moritz Scherer, ETH Zurich
 * - Philip Wiese, ETH Zurich
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

#include "DeeployBasicMath.h"

#include <stdlib.h>

/**
 * @todo Remove malloc in function and make the buffer a pointer passed to the
 * fuction which is allocated by the user.
 */
void Softmax_s8_s8(int8_t *data_in, int8_t *data_out, uint32_t size,
                   uint32_t lastDimLength, int32_t coeffA, int32_t coeffB,
                   int64_t coeffC, int32_t log2, uint32_t n_levels) {

  int8_t z;
  int16_t xTilde, p;
  uint32_t y_sum;
  int8_t x_max;
  uint32_t *y = (uint32_t *)deeploy_malloc(sizeof(int32_t) * lastDimLength);

  for (uint32_t i = 0; i < size / lastDimLength; i++) {
    y_sum = 0;
    x_max = -128;
    for (uint32_t j = 0; j < lastDimLength; j++) {
      if (data_in[j + i * lastDimLength] > x_max) {
        x_max = data_in[j + i * lastDimLength];
      }
    }
    for (uint32_t j = 0; j < lastDimLength; j++) {
      xTilde = (data_in[j + i * lastDimLength] - x_max);
      z = (int8_t)(-(xTilde / log2));
      z = CLAMP(z, 0, 31);
      p = (int16_t)(xTilde + z * log2);
      y[j] = (uint32_t)((uint64_t)(coeffA * ((p + coeffB) * (p + coeffB)) +
                                   coeffC) >>
                        (z));
      y_sum += y[j];
    }
    for (uint32_t j = 0; j < lastDimLength; j++) {
      data_out[j + i * lastDimLength] =
          (int8_t)((y[j] * (n_levels - 1)) / (y_sum)-n_levels / 2);
    }
  }
  deeploy_free(y);
}

void ITAMax_s8(int8_t const *__restrict__ pSrcA, int8_t *__restrict__ pDstB,
               int8_t *__restrict__ pBufN, uint32_t size,
               uint32_t lastDimLength, uint32_t n_levels) {

  uint32_t i = 0; // Row Counter
  uint32_t j = 0; // Column Counter

  uint8_t *shift = (uint8_t *)pBufN;

  for (i = 0; i < size / lastDimLength; ++i) {
    // 1. Find maximum over row
    int8_t max = -128;
    for (j = 0; j < lastDimLength; ++j) {
      if (pSrcA[i * lastDimLength + j] > max) {
        max = pSrcA[i * lastDimLength + j];
      }
    }

    // 2. Calculate exponential sum
    uint32_t exp_sum = 0;
    for (j = 0; j < lastDimLength; ++j) {
      int32_t diff = max - pSrcA[i * lastDimLength + j];
      shift[j] = (uint8_t)((diff + 16) >> 5);
      exp_sum += (256U >> shift[j]);
    }

    uint32_t exp_sum_inv = ((n_levels - 1) * 256U) / exp_sum;

    for (j = 0; j < lastDimLength; ++j) {
      pDstB[i * lastDimLength + j] =
          (int8_t)((exp_sum_inv >> shift[j]) - (n_levels / 2));
    }
  }
}

void ITAPartialMax_s8(int8_t const *__restrict__ pSrcA,
                      int8_t *__restrict__ pDstB, uint32_t size,
                      uint32_t lastDimLength, uint32_t group_width,
                      uint32_t n_levels) {

  uint32_t i = 0; // Row Counter
  uint32_t j = 0; // Column Counter
  uint32_t g = 0; // Group Counter

  // Iterate over rows
  for (i = 0; i < size / lastDimLength; ++i) {

    // Initialize denominator
    uint32_t exp_partial_sum = 0;

    // Initialize maximum with minimal possible value
    int8_t global_max = -128;

    // STAGE 1: Compute the denominator of the softmax
    // Iterate over groups
    for (g = 0; g < lastDimLength / group_width; ++g) {

      // Find the maximum for each row in the current column block
      int8_t current_max = -128;
      for (uint32_t k = 0; k < group_width; ++k) {
        int8_t value = pSrcA[i * lastDimLength + g * group_width + k];
        if (value > current_max) {
          current_max = value;
        }
      }

      // Calculate shift values (integer division with rounding)
      int32_t max_shift = (current_max - global_max + 16) >> 5;

      // Update all shift values where new maximum is larger
      int32_t shift_sum = (current_max > global_max) ? max_shift : 0;
      global_max = (current_max > global_max) ? current_max : global_max;

      // Calculate exponential sum over the current part of the row
      uint32_t exp_sum = 0;
      for (uint32_t k = 0; k < group_width; ++k) {
        int32_t diff =
            global_max - pSrcA[i * lastDimLength + g * group_width + k];
        uint8_t shift = (uint8_t)((diff + 16) >> 5);
        exp_sum += (256U >> shift);
      }

      // Update the accumulated sum and add the accumulation over the current
      // part of the row
      exp_partial_sum = (exp_partial_sum >> shift_sum) + exp_sum;

      // deeploy_log("[R %d,G %d]: %6d, %6d, %6d, %6d, %6d, %6d\n", i, g,
      // current_max, max_shift, shift_sum, global_max, exp_sum,
      // exp_partial_sum);
    }

    // STAGE 2: Calculate the softmax activation
    // WIESEP: Scale Softmax to 127
    // The Softmax values are maximum 127 as sumdot modules can only do
    // signed-signed operations for now. This is a temporary fix until sumdot is
    // fixed.
    uint32_t exp_partial_sum_inverse =
        ((n_levels / 2 - 1) * 256U) / exp_partial_sum;

    for (j = 0; j < lastDimLength; ++j) {
      // Find the difference between the maximum and x
      int32_t diff = global_max - pSrcA[i * lastDimLength + j];

      // Shift the values by B-log2B -> multiply by B/2**B = log2e*eps_x
      // (integer division with rounding)
      uint8_t shift = (uint8_t)((diff + 16) >> 5);

      // Calculate the activation value
      pDstB[i * lastDimLength + j] =
          (int8_t)((exp_partial_sum_inverse >> shift) - (n_levels / 2));

      // deeploy_log("[R %d,C %d]: %6d, %6d, %6d, %6d, %6d\n", i, j, pSrcA[i *
      // lastDimLength + j], diff, shift, exp_partial_sum_inverse, pDstB[i *
      // lastDimLength + j]);
    }
  }
}