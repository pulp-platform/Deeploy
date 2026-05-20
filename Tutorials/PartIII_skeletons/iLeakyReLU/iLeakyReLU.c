/* =====================================================================
 * Title:        iLeakyReLU.c  (SoCDAML Part III - Step 3 skeleton)
 *
 * Plain-C int8 LeakyReLU. The per-core chunking boilerplate is provided.
 * Fill in the inner loop body marked `TODO(student)`.
 *
 *   Goal: out[i] = (in[i] >= 0) ? in[i] : ((mul * in[i]) >> shift)
 *
 * Hints:
 *   - Cast in[i] to int32_t before the multiply to avoid 8-bit overflow.
 *   - Cast the final result back to int8_t before storing.
 *
 * Drop into: TargetLibraries/PULPOpen/src/iLeakyReLU.c
 * ===================================================================== */
/* SPDX-License-Identifier: Apache-2.0 */

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
    // TODO(student): compute pOut[i] from pIn[i], mul, shift.
    // Replace the following line:
    pOut[i] = 0;
  }
}
