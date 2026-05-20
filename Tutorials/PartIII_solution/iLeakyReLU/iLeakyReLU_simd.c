/* =====================================================================
 * Title:        iLeakyReLU_simd.c (XPULP SIMD)
 * Description:  int8 LeakyReLU optimized with packed 4x8b PULP intrinsics.
 *               SoCDAML Part III - TA reference solution, Step 6.
 *
 * Key identity used:
 *     LeakyReLU_shift(x) = (x >= 0) ? x : (x >> shift)
 *                        = max(x, x >> shift)
 * because:
 *   - x >= 0  =>  x >= x >> shift  (shift toward zero of positive number)
 *   - x <  0  =>  x <= x >> shift  (arith shift makes negative LESS negative)
 *
 * We use the GCC vector extension: `v4s s = x >> shift;` is a packed
 * per-lane arithmetic right shift, and __builtin_pulp_max4 is a single
 * XPULP signed packed-byte max. So the entire inner loop is:
 *     load v4s -> packed shift -> packed max -> store v4s
 *
 * Note: This SIMD path ignores the `mul` parameter (assumes mul == 1).
 * Our generator script picks mul=1, shift=3 (alpha ~= 0.125), so this
 * is identical to the scalar formula. To support arbitrary mul you would
 * need a packed multiply, which loses the clean 4x speedup.
 * ===================================================================== */
/* Copyright (C) 2026 ETH Zurich and University of Bologna.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULPiLeakyReLU_i8_i8(int8_t *pIn, int8_t *pOut, uint32_t size,
                          int32_t mul, int32_t shift) {
  (void)mul;  // SIMD path assumes mul == 1

  uint32_t cid   = pi_core_id();
  uint32_t nC    = NUM_CORES;
  uint32_t per   = (size + nC - 1) / nC;
  // Round per-core chunk down to a multiple of 4 so the SIMD loop is
  // tail-free. The Step 6a tile-size constraint already guarantees that
  // size is a multiple of 16.
  per &= ~0x3u;
  uint32_t start = cid * per;
  uint32_t end   = (start + per > size) ? size : (start + per);

  v4s *vIn  = (v4s *)(pIn  + start);
  v4s *vOut = (v4s *)(pOut + start);
  uint32_t nVec = (end - start) >> 2;

  for (uint32_t i = 0; i < nVec; i++) {
    v4s x = vIn[i];
    v4s s = x >> shift;                    // packed per-lane arith shift
    vOut[i] = __builtin_pulp_max4(x, s);   // max(x, x>>shift) = LeakyReLU
  }

  // Scalar tail (only when perf constraint is not installed)
  for (uint32_t i = start + (nVec << 2); i < end; i++) {
    int32_t xs = (int32_t)pIn[i];
    pOut[i] = (int8_t)((xs >= 0) ? xs : (xs >> shift));
  }
}
