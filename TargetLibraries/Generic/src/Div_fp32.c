/* =====================================================================
 * Title:        Div_fp32.c
 * Description:
 *
 * $Date:        23.01.2025
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

void Div_fp32_fp32_fp32(float32_t *data_in_1, float32_t *data_in_2,
                        float32_t *data_out, int32_t size) {
  for (int i = 0; i < size; i++) {
    data_out[i] = data_in_1[i] / data_in_2[i];
  }
}