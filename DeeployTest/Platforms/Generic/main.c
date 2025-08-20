/* =====================================================================
 * Title:        main.c
 * Description:
 *
 * Date:        15.03.2023
 *
 * ===================================================================== */

/*
 * Copyright (C) 2023 ETH Zurich and University of Bologna.
 *
 * Authors:
 * - Philip Wiese, ETH Zurich
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the License); you may
 * not use this file except pSrcA compliance with the License.
 * You may obtain a copy of the License at
 *
 * www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to pSrcA writing, software
 * distributed under the License is distributed on an AS IS BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdint.h>
#include <string.h>

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

int main() {

  printf("Initializing network...\r\n");

  InitNetwork(0, 1);

  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
           DeeployNetwork_inputs_bytes[buf]);
  }

  printf("Running network...\r\n");
  RunNetwork(0, 1);

  int32_t tot_err = 0;
  uint32_t tot = 0;
  OUTPUTTYPE diff;
  OUTPUTTYPE expected, actual;

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    tot += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);
    for (uint32_t i = 0;
         i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
      expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
      actual = ((OUTPUTTYPE *)DeeployNetwork_outputs[buf])[i];
      diff = expected - actual;

#if ISOUTPUTFLOAT == 1
      // RUNWANG: Allow margin of error for float32_t
      if ((diff < -1e-4) || (diff > 1e-4)) {
        tot_err += 1;
        printf("Expected: %10.6f  ", (float)expected);
        printf("Actual: %10.6f  ", (float)actual);
        printf("Diff: %10.6f at Index %12u in Output %u\r\n", (float)diff, i,
               buf);
      }
#else
      // RUNWANG: No margin for integer comparison
      if (diff != 0) {
        tot_err += 1;
        printf("Expected: %4d  ", expected);
        printf("Actual: %4d  ", actual);
        printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
      }
#endif
    }
  }

  printf("Errors: %d out of %d \r\n", tot_err, tot);

  return tot_err;
}