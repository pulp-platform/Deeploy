/* =====================================================================
 * Title:        iLeakyReLU_simd.c  (SoCDAML Part III - Step 6b skeleton)
 *
 * SIMD version of iLeakyReLU using XPULP packed 4x8b operations.
 * The per-core chunking is provided. Fill in the inner SIMD body.
 *
 * Key identity (worth deriving on paper before reading hints below):
 *     LeakyReLU(x) = (x >= 0) ? x : (x >> shift)
 *                  = max(x, x >> shift)
 * because arithmetic right shift makes a negative value LESS negative
 * (or zero) and doesn't change the sign of a non-negative value.
 *
 * Strategy hint (one path, two intrinsic-level operations per 4 lanes):
 *   - load v4s lane:        v4s x = vIn[i];
 *   - per-lane signed shift: v4s s = x >> shift;       (GCC vector ext)
 *   - signed packed max:    __builtin_pulp_max4(x, s);
 *
 * For the lab we assume `mul == 1` (the generator picks mul=1, shift=3).
 *
 * Drop into: TargetLibraries/PULPOpen/src/iLeakyReLU.c (overwrite scalar)
 * ===================================================================== */
/* SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULPiLeakyReLU_i8_i8(int8_t *pIn, int8_t *pOut, uint32_t size, int32_t mul,
                          int32_t shift) {
  (void)mul; // SIMD path assumes mul == 1

  uint32_t cid = pi_core_id();
  uint32_t nC = NUM_CORES;

  // Whole 4-element vectors are split across the cores by vector index, so
  // that no element is lost when size / nC is small or size is not a
  // multiple of 4 * nC.
  uint32_t nVec = size >> 2;
  uint32_t perVec = (nVec + nC - 1) / nC;
  uint32_t vStart = cid * perVec;
  uint32_t vEnd = (vStart + perVec > nVec) ? nVec : (vStart + perVec);

  v4s *vIn = (v4s *)pIn;
  v4s *vOut = (v4s *)pOut;

  for (uint32_t i = vStart; i < vEnd; i++) {
    v4s x = vIn[i];
    // TODO(student): one line to compute `s` from `x` and `shift`,
    //                one line to blend `x` and `s` with the packed
    //                signed max intrinsic and store it.
    vOut[i] = x; // <- placeholder, replace
  }

  // The trailing size % 4 elements never fill a vector; one core handles
  // them. Disjoint from every vector chunk above, so no sync is needed.
  if (cid == 0) {
    for (uint32_t i = nVec << 2; i < size; i++) {
      int32_t xs = (int32_t)pIn[i];
      pOut[i] = (int8_t)((xs >= 0) ? xs : (xs >> shift));
    }
  }
}
