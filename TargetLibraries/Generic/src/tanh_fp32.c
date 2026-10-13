/*
 * SPDX-FileCopyrightText: 2026 EPFL, ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployBasicMath.h"
#include <math.h>

void tanh_fp32(float32_t const *__restrict__ pSrcA, uint32_t size,
               float32_t *__restrict__ pDstC) {
  for (uint32_t i = 0; i < size; i++) {
    pDstC[i] = tanhf(pSrcA[i]);
  }
}