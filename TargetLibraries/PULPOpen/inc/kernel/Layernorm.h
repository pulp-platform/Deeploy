/* =====================================================================
 * Title:        Layernorm.h
 * Description:
 *
 * $Date:        05.06.2025
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

void PULP_Layernorm_fp32_fp32(float32_t *data_in, float32_t *data_out,
                              float32_t *scale, float32_t *bias,
                              float32_t epsilon, uint32_t size,
                              uint32_t lastDimLength);