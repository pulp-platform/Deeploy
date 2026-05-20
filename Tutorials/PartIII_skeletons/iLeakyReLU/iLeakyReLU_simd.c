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
/* SPDX-License-Identifier: Apache-2.0 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

void PULPiLeakyReLU_i8_i8(int8_t *pIn, int8_t *pOut, uint32_t size,
                          int32_t mul, int32_t shift) {
  (void)mul;  // SIMD path assumes mul == 1

  uint32_t cid   = pi_core_id();
  uint32_t nC    = NUM_CORES;
  uint32_t per   = (size + nC - 1) / nC;
  per &= ~0x3u;
  uint32_t start = cid * per;
  uint32_t end   = (start + per > size) ? size : (start + per);

  v4s *vIn  = (v4s *)(pIn  + start);
  v4s *vOut = (v4s *)(pOut + start);
  uint32_t nVec = (end - start) >> 2;

  for (uint32_t i = 0; i < nVec; i++) {
    v4s x = vIn[i];
    // TODO(student): one line to compute `s` from `x` and `shift`,
    //                one line to blend `x` and `s` with the packed
    //                signed max intrinsic and store it.
    vOut[i] = x;  // <- placeholder, replace
  }
}
