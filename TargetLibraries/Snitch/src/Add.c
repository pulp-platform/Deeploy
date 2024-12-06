/* ----------------------------------------------------------------------
#
# File: Add.c
#
# Last edited: 11.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
*/

#include "DeeploySnitchMath.h"

void SnitchAdd(int8_t *pIn1, int8_t *pIn2, int32_t *pOut, uint32_t size,
               int32_t offset) {

  uint32_t core_id = snrt_global_compute_core_idx();
  uint32_t numThreads = snrt_global_compute_core_num();
  uint32_t chunk, chunkSize, start, stop;

  chunkSize = size / numThreads;
  if (core_id < (numThreads - 1)) {
    chunk = chunkSize * core_id;
    stop = chunk + chunkSize;
  } else {
    chunk = (chunkSize * core_id - 1) + (size - chunk);
    stop = size;
  }
  start = chunk;

#pragma loopunroll 2
  for (int i = start; i < stop; i++) {
    pOut[i] = pIn1[i] + pIn2[i] + offset;
  }
}
