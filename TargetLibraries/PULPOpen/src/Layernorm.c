
/* =====================================================================
 * Title:        Layernorm.c
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

void PULP_Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias,
                              float32_t epsilon, uint32_t size,
                              uint32_t lastDimLength) {

  int8_t core_id = pi_core_id();
  int8_t log2Core = log2(NUM_CORES);

  int32_t seq_length = size / lastDimLength;
  int32_t chunk =
      (seq_length >> log2Core) + ((seq_length & (NUM_CORES - 1)) != 0);
  int32_t start_seq = MIN(chunk * core_id, seq_length);
  int32_t end_seq = MIN(start_seq + chunk, seq_length);

  int32_t elem_start = start_seq * lastDimLength;
  int32_t elem_end = end_seq * lastDimLength;

  float32_t *local_data_in = data_in + elem_start;
  float32_t *local_data_out = data_out + elem_start;
  int32_t local_size = elem_end - elem_start;

  float32_t mean;
  float32_t sum;
  float32_t std;
  float32_t temp;

  int32_t local_seq_count = local_size / lastDimLength;

  for (int32_t i = 0; i < local_seq_count; i++) {

    sum = 0.0f;
    mean = 0.0f;
    for (int32_t j = 0; j < lastDimLength; j++) {
      mean += local_data_in[j + i * lastDimLength];
    }
    mean = mean / (float32_t)lastDimLength;

    sum = 0.0f;
    for (int32_t j = 0; j < lastDimLength; j++) {
      temp = local_data_in[j + i * lastDimLength] - mean;
      sum += temp * temp;
    }
    sum = sum / (float32_t)lastDimLength;
    sum += epsilon;
    std = sqrtf(sum);

    for (int32_t j = 0; j < lastDimLength; j++) {
      local_data_out[j + i * lastDimLength] =
          ((local_data_in[j + i * lastDimLength] - mean) / std) * scale[j] +
          bias[j];
    }
  }
}