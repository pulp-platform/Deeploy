/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
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

void PULP_Gemm_fp32_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
                                   const float32_t *__restrict__ pSrcB,
                                   const float32_t *__restrict__ pDstC,
                                   float32_t *__restrict__ pDstY, uint32_t M,
                                   uint32_t N, uint32_t O, uint32_t transA,
                                   uint32_t transB);

#endif // __DEEPLOY_MATH_GEMM_KERNEL_HEADER_