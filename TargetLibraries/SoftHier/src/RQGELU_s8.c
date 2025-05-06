/* =====================================================================
 * Title:        RQGELU_s8.c
 * Description:
 *
 * $Date:        19.12.2022
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

void RQGELU_s8_s8(int8_t *data_in, int8_t *data_out, int32_t dataSize, int8_t b,
                  int16_t one, int32_t input_offset, int32_t output_offset,
                  int32_t *mul, int32_t *add, int32_t *shift) {

  int32_t sign, x, x_abs, q;
  int32_t d;
  int32_t L, y;
  int32_t intermediate;

  for (int i = 0; i < dataSize; i++) {
    x = data_in[i] + input_offset;
    sign = (x > 0) - (x < 0); // sgn(x)
    x_abs = sign * x;         // abs(x)
    if (x_abs > -b) {
      q = -b;
    } else {
      q = x_abs;
    }
    d = q + b;
    L = sign * (-(d * d) + one);
    y = x * (((one + L)) >> 1);

    intermediate = ((int32_t)y) * (*mul) + (*add);
    intermediate =
        ((intermediate + ((1 << ((*shift) - 1)))) >> (*shift)) + output_offset;
    data_out[i] = (int8_t)CLAMP(intermediate, -128, 127);
  }
}
