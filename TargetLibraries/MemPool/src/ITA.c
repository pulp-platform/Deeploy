/* =====================================================================
 * Title:        ITA.c
 * Description:
 *
 * Date:         5.12.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
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

#include "DeeployMath.h"

void ITA_getStruct(ita_data_t *ita_data, int8_t *base_address, uint32_t S,
                   uint32_t E, uint32_t P) {
  ita_data->wo_weight = base_address;
  ita_data->wv_weight = ita_data->wo_weight + E * P;
  ita_data->wk_weight = ita_data->wv_weight + E * P;
  ita_data->q = ita_data->wk_weight + E * P;
  ita_data->k = ita_data->q + S * E;
  ita_data->wq_weight = ita_data->k + S * E;
  ita_data->wo_bias = (int32_t *)ita_data->wq_weight + E * P; // 32 bit values
  ita_data->wv_bias = ita_data->wo_bias + 1 * E;              // 32 bit values
  ita_data->wk_bias = ita_data->wv_bias + 1 * P;              // 32 bit values
  ita_data->wq_bias = ita_data->wk_bias + 1 * P;              // 32 bit values
}

// The tensors have to be stored in a split layout equivalent to
// np.reshape(np.concatenate(np.split(self.Q, self.split, axis = 1)),
// (self.S_ITA, self.E_ITA))
void ITA_copyInput(int8_t *pDst, int8_t const *__restrict__ pSrc, uint32_t S,
                   uint32_t E, int8_t offset) {
  uint32_t i = 0;
  uint32_t j = 0;
  uint32_t k = 0;

  for (i = 0; i < E / ITA_PE; ++i) {
    for (j = 0; j < S; ++j) {
      if (offset != 0) {
        for (k = 0; k < ITA_PE; ++k) {
          pDst[i * S * ITA_PE + j * ITA_PE + k] =
              (int8_t)(pSrc[i * ITA_PE + j * E + k] + offset);
        }
      } else {
#if USE_DMA
        dma_memcpy_blocking((void *)&pDst[i * S * ITA_PE + j * ITA_PE],
                            (void *)&pSrc[i * ITA_PE + j * E], ITA_PE);
#else
        memcpy((void *)&data[i * S * ITA_PE + j * ITA_PE],
               (void *)&pSrc[i * ITA_PE + j * E], ITA_PE);
#endif
      }
    }
  }
}

void ITA_printAddresses(ita_data_t *ita_data) {
  deeploy_log("ITA addresses:\n");
  deeploy_log("wo_weight: %p\n", ita_data->wo_weight);
  deeploy_log("wv_weight: %p\n", ita_data->wv_weight);
  deeploy_log("wk_weight: %p\n", ita_data->wk_weight);
  deeploy_log("q: %p\n", ita_data->q);
  deeploy_log("k: %p\n", ita_data->k);
  deeploy_log("wq_weight: %p\n", ita_data->wq_weight);
  deeploy_log("wo_bias: %p\n", ita_data->wo_bias);
  deeploy_log("wv_bias: %p\n", ita_data->wv_bias);
  deeploy_log("wk_bias: %p\n", ita_data->wk_bias);
  deeploy_log("wq_bias: %p\n", ita_data->wq_bias);
}