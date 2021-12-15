/* =====================================================================
 * Title:        vec2mat.h
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

#include "stdint.h"

void gemv_s8_s8_plp(int8_t *pIn, int8_t *pBias, int8_t *pOut, int8_t *pWeight,
                    int32_t *pKappa, int32_t *pLambda, uint16_t out_mult,
                    uint16_t out_shift, uint16_t dim_vec,
                    uint16_t num_o_neurons, uint8_t flag_relu,
                    uint8_t flag_batch_norm);
