/* =====================================================================
 * Title:        RQiHardswish.c
 * Description:
 *
 * $Date:        15.03.2024
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

#include "DeeployPULPMath.h"

void RQiHardswish_s8_s8_plp(int8_t *input, int8_t *output, int32_t size,
                            int32_t one_over_six, int32_t three, int32_t six,
                            int32_t mul, int32_t add, int32_t shift) {

  int32_t temp;
  int32_t rnd;

  rnd = (1 << (shift - 1));

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);
  int16_t chunk = (size >> log2Core) + ((size & (NUM_CORES - 1)) != 0);
  int16_t chunk_start = MIN(chunk * core_id, size);
  int16_t chunk_stop = MIN(chunk_start + chunk, size + 1);

#pragma unroll 2
  for (int i = chunk_start; i < chunk_stop; i++) {
    temp = input[i] + three;
    temp = CLAMP(temp, 0, six);

    temp = temp * one_over_six;
    temp = input[i] * temp;
    temp = temp * (mul) + (add + rnd);

    temp = temp >> shift;

    output[i] = (int8_t)CLAMP(temp, -128, 127);
  }
}
