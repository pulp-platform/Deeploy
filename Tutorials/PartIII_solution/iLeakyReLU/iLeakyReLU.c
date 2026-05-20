/* =====================================================================
 * Title:        iLeakyReLU.c (scalar baseline)
 * Description:  int8 quantization-friendly LeakyReLU, plain C.
 *               SoCDAML Part III - TA reference solution, Step 3.
 * ===================================================================== */
/* Copyright (C) 2026 ETH Zurich and University of Bologna.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULPiLeakyReLU_i8_i8(int8_t *pIn, int8_t *pOut, uint32_t size,
                          int32_t mul, int32_t shift) {
  uint32_t cid   = pi_core_id();
  uint32_t nC    = NUM_CORES;
  uint32_t per   = (size + nC - 1) / nC;
  uint32_t start = cid * per;
  uint32_t end   = (start + per > size) ? size : (start + per);

  for (uint32_t i = start; i < end; i++) {
    int32_t x  = (int32_t)pIn[i];
    int32_t lo = (mul * x) >> shift;
    pOut[i]    = (int8_t)((x >= 0) ? x : lo);
  }
}
