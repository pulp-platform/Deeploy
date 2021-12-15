/* =====================================================================
 * Title:        iRMSNorm_s8.c
 * Description:
 *
 * $Date:        20.02.2024
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
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

void _plp_sqrt_q32(const int32_t *__restrict__ pSrc, const uint32_t fracBits,
                   int32_t *__restrict__ pRes) {

  int32_t number = *pSrc;
  int32_t root = 0;

  int32_t start = 0;
  int32_t end = 46342; // smallest integer that is larger than sqrt(0x7FFFFFFF)
  int32_t mid;

  if (number > 0) {

    while (start <= end) {

      mid = (start + end) >> 1;

      if (((mid * mid) >> fracBits) == number) {
        root = mid;
        break;
      }

      if (((mid * mid) >> fracBits) < number) {
        start = mid + 1;
        root = mid;
      } else {
        end = mid - 1;
      }
    }

    *pRes = root;

  } else {
    *pRes = 0;
  }
}

void iRMSnorm_s8_s8(int8_t *data_in, int8_t *data_out, int32_t *weight,
                    int32_t input_offset, int32_t size, int32_t lastDimLength,
                    int32_t log2D) {

  // int16_t temp[size];
  int32_t sum;
  int32_t std;
  int16_t temp;
  int32_t intermediate;

  for (int i = 0; i < (size / lastDimLength); i++) {
    sum = 0;
    for (int j = 0; j < lastDimLength; j++) {
      temp = (int16_t)(data_in[j + i * lastDimLength] + input_offset);
      sum += temp * temp;
    }
    sum = sum / lastDimLength;
    sum += 1;
    _plp_sqrt_q32(&sum, 0, &std);

    for (int j = 0; j < lastDimLength; j++) {

      intermediate =
          ((((((int32_t)data_in[j + i * lastDimLength]) + input_offset) *
             weight[j]) /
            (std)) >>
           log2D);

      data_out[j + i * lastDimLength] = (int8_t)CLAMP(intermediate, -128, 127);
    }
  }
}
