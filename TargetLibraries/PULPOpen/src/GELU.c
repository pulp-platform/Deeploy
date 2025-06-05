/* =====================================================================
 * Title:        GELU.c
 * Description:  
 *
 * $Date:        05.06.2025
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


#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"

#include "DeeployPULPMath.h"

#define M_PI 3.14159265358979323846

void PULP_GELU_fp32_fp32(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
    int8_t core_id = pi_core_id();
    int8_t log2Core = log2(NUM_CORES);
    int16_t chunk = (dataSize >> log2Core) + ((dataSize & (NUM_CORES-1))!=0);
    int16_t chunk_start = MIN(chunk * core_id, dataSize);
    int16_t chunk_stop = MIN(chunk_start + chunk, dataSize);
    const float32_t sqrt_2_over_pi = 0.7978845608f; // sqrt(2/π)
    const float32_t coeff = 0.044715f;

    for (uint32_t i = chunk_start; i < chunk_stop; i++) {
        float32_t x = data_in[i];
        float32_t x_cubed = x * x * x;
        float32_t inner = sqrt_2_over_pi * (x + coeff * x_cubed);
        
        float32_t exp_2z = expf(2.0f * inner);
        float32_t tanh_val = (exp_2z - 1.0f) / (exp_2z + 1.0f);
        
        float32_t cdf = 0.5f * (1.0f + tanh_val);
        data_out[i] = x * cdf;
    }
}

void PULP_GELU_fp32_fp32_sigmoid(float32_t *data_in, float32_t *data_out, int32_t dataSize) {
    int8_t core_id = pi_core_id();
    int8_t log2Core = log2(NUM_CORES);
    int16_t chunk = (dataSize >> log2Core) + ((dataSize & (NUM_CORES-1))!=0);
    int16_t chunk_start = MIN(chunk * core_id, dataSize);
    int16_t chunk_stop = MIN(chunk_start + chunk, dataSize);
    
    const float32_t scale = 1.702f;
    for (uint32_t i = chunk_start; i < chunk_stop; i++) {
        float32_t x = data_in[i];
        float32_t sigmoid_in = scale * x;
        // sigmoid(z) = 1 / (1 + exp(-z))
        float32_t sigmoid = 1.0f / (1.0f + expf(-sigmoid_in));
        data_out[i] = x * sigmoid;
    }
}