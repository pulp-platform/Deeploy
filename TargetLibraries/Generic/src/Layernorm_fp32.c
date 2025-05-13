/* =====================================================================
 * Title:        Layernorm_fp32.c
 * Description:
 *
 * $Date:        22.01.2025
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Run Wang, ETH Zurich
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

void Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                         float32_t *scale, float32_t *bias, float32_t epsilon,
                         int32_t size, int32_t lastDimLength) {
  float32_t mean;
  float32_t sum;
  float32_t std;
  float32_t temp;

  for (int i = 0; i < (size / lastDimLength); i++) {
    sum = 0.0f;
    mean = 0.0f;
    for (int j = 0; j < lastDimLength; j++) {
      mean += data_in[j + i * lastDimLength];
    }
    mean = mean / lastDimLength;
    for (int j = 0; j < lastDimLength; j++) {
      temp = data_in[j + i * lastDimLength] - mean;
      sum += temp * temp;
    }
    sum = sum / lastDimLength;
    sum += epsilon;
    std = sqrtf(sum);

    for (int j = 0; j < lastDimLength; j++) {
      data_out[j + i * lastDimLength] =
          ((data_in[j + i * lastDimLength] - mean) / std) * scale[j] + bias[j];
    }
  }
}