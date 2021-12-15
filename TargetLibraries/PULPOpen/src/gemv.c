/* =====================================================================
 * Title:        vec2mat.c
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

#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"

#include "DeeployPULPMath.h"

void gemv_s8_s8_plp(int8_t *pIn, int8_t *pBias, int8_t *pOut, int8_t *pWeight,
                    int32_t *pKappa, int32_t *pLambda, uint16_t out_mult,
                    uint16_t out_shift, uint16_t dim_vec,
                    uint16_t num_o_neurons, uint8_t flag_relu,
                    uint8_t flag_batch_norm) {

  uint16_t dim_vec_in = dim_vec;
  uint16_t dim_vec_wt = dim_vec;

  int start = 0;
  int stop = num_o_neurons;

  v4s vecA;
  v4s vecB;
  v4s vecB2;

  int8_t *pOutBuffer = (int8_t *)pOut + start;
  int lft_neurons = num_o_neurons & 0x01;
  int stop_even = stop - lft_neurons;

  int i;
  int32_t *k1 = pKappa + start;
  int32_t *lambda1 = pLambda + start;

  for (i = start; i < stop_even; i += 2) {
    int sum = 0;
    int sum2 = 0;
    if (pBias != NULL) {
      sum = *(int32_t *)(pBias + 4 * i);
      sum2 = *(int32_t *)(pBias + 4 * i + 4);
    }

    int8_t *pA = pIn;
    int8_t *pB = pWeight + (i * dim_vec_wt);
    int8_t *pB2 = pB + dim_vec_wt;

    for (int j = 0; j < (dim_vec >> 2); j++) {
      vecA = *((v4s *)pA);
      vecB = *((v4s *)pB);
      vecB2 = *((v4s *)pB2);
      sum = SumDotps4(vecA, vecB, sum);
      sum2 = SumDotps4(vecA, vecB2, sum2);
      pA += 4;
      pB += 4;
      pB2 += 4;
    }
    uint16_t col_cnt = dim_vec & 0x3;
    while (col_cnt) {
      int8_t inA = *pA;
      pA++;
      int8_t inB = *pB;
      pB++;
      int8_t inB2 = *pB2;
      pB2++;
      sum += inA * inB;
      sum2 += inA * inB2;
      col_cnt--;
    }
    if (flag_batch_norm && flag_relu) {
      *pOutBuffer = pulp_nn_bn_quant_i8(sum, *k1, *lambda1, out_shift);
      pOutBuffer++;
      *pOutBuffer =
          pulp_nn_bn_quant_i8(sum2, *(k1 + 1), *(lambda1 + 1), out_shift);
      pOutBuffer++;
      k1 += 2;
      lambda1 += 2;
    } else {
      if (flag_relu == 1) {
        *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
        pOutBuffer++;
        *pOutBuffer = pulp_nn_quant_i8(sum2, out_mult, out_shift);
        pOutBuffer++;
      } else {
        *pOutBuffer = (int8_t)clips8(sum >> out_shift);
        pOutBuffer++;
        *pOutBuffer = (int8_t)clips8(sum2 >> out_shift);
        pOutBuffer++;
      }
    }
  }
  if (lft_neurons && (stop - start) > 0) {
    int sum = 0;
    if (pBias != NULL) {
      sum = *(int32_t *)(pBias + 4 * i);
    }

    int8_t *pA = pIn;
    int8_t *pB = pWeight + (i * dim_vec_wt);

    for (int j = 0; j < (dim_vec >> 2); j++) {
      vecA = *((v4s *)pA);
      vecB = *((v4s *)pB);
      sum = SumDotps4(vecA, vecB, sum);
      pA += 4;
      pB += 4;
    }
    uint16_t col_cnt = dim_vec & 0x3;
    while (col_cnt) {
      int8_t inA = *pA;
      pA++;
      int8_t inB = *pB;
      pB++;
      sum += inA * inB;
      col_cnt--;
    }
    if (flag_batch_norm && flag_relu) {
      *pOutBuffer = pulp_nn_bn_quant_i8(sum, *pKappa, *pLambda, out_shift);
      pOutBuffer++;
      pKappa++;
      pLambda++;
    } else {
      if (flag_relu == 1) {
        *pOutBuffer = pulp_nn_quant_i8(sum, out_mult, out_shift);
        pOutBuffer++;
      } else {
        *pOutBuffer = (int8_t)clips8(sum >> out_shift);
        pOutBuffer++;
      }
    }
  }
  pi_cl_team_barrier(0);
}
