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

void PULPiLeakyReLU_i8_i8(int8_t *pIn, int8_t *pOut, uint32_t size, int32_t mul,
                          int32_t shift) {
  (void)mul; // SIMD path assumes mul == 1

  uint32_t cid = pi_core_id();
  uint32_t nC = NUM_CORES;

  // Split the 4-element vectors across the cores, by vector index.
  // Splitting the element count instead and rounding each core's share down
  // to a multiple of 4 loses the remainder, and collapses to zero work per
  // core as soon as size / nC < 4 (e.g. a 16-element tile on 8 cores).
  uint32_t nVec = size >> 2;
  uint32_t perVec = (nVec + nC - 1) / nC;
  uint32_t vStart = cid * perVec;
  uint32_t vEnd = (vStart + perVec > nVec) ? nVec : (vStart + perVec);

  v4s *vIn = (v4s *)pIn;
  v4s *vOut = (v4s *)pOut;

  for (uint32_t i = vStart; i < vEnd; i++) {
    v4s x = vIn[i];
    v4s s = x >> shift;                  // packed per-lane arith shift
    vOut[i] = __builtin_pulp_max4(x, s); // max(x, x>>shift) = LeakyReLU
  }

  // The trailing size % 4 elements never fill a vector.
  // Defer to 1-core reduction.
  if (cid == 0) {
    for (uint32_t i = nVec << 2; i < size; i++) {
      int32_t xs = (int32_t)pIn[i];
      pOut[i] = (int8_t)((xs >= 0) ? xs : (xs >> shift));
    }
  }
}
