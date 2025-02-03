/* =====================================================================
 * Title:        GELU_fp32.c
 * Description:
 *
 * $Date:        19.12.2022
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
#define M_PI 3.14159265358979323846

void GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
   for (int i = 0; i < dataSize; i++) {
        float32_t x = data_in[i];
        float32_t cdf = 0.5 * (1.0 + tanh((sqrt(2 / M_PI) * (x + 0.044715 * pow(x, 3)))));
        data_out[i] = x * cdf;
    }
}
