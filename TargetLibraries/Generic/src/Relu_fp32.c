/* =====================================================================
 * Title:        Softmax_fp8.c
 * Description:
 *
 * $Date:        22.01.2025
 *
 * ===================================================================== */
/*
 * Copyright (C) 2022 ETH Zurich and University of Bologna.
 *
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


void Relu_fp32_fp32(float32_t* input, float32_t* output, int32_t size, int32_t last_dim_length) {

    int32_t batch_size = size / last_dim_length;  

    for (int b = 0; b < batch_size; b++) {
        for (int i = 0; i < last_dim_length; i++) {
            output[b * last_dim_length + i] = MAX(input[b * last_dim_length + i], 0.0f);
        }
    }
}