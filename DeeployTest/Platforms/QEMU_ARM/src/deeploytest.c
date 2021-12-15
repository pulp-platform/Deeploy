/* =====================================================================
 * Title:        deeploytest.c
 * Description:
 *
 * Date:        15.03.2023
 *
 * ===================================================================== */
/*
 * Copyright (C) 2020 ETH Zurich and University of Bologna.
 *
 * Author: Moritz Scherer, ETH Zurich
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

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"
#include <stdint.h>
#include <stdlib.h>

int main(void) {
  InitNetwork(0, 1);

  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
           DeeployNetwork_inputs_bytes[buf]);
  }

  RunNetwork(0, 1);

  int32_t tot_err = 0;
  uint32_t tot = 0;
  int32_t diff;
  int32_t expected, actual;

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    tot += DeeployNetwork_outputs_bytes[buf];
    for (uint32_t i = 0; i < DeeployNetwork_outputs_bytes[buf]; i++) {
      expected = ((char *)testOutputVector[buf])[i];
      actual = ((char *)DeeployNetwork_outputs[buf])[i];
      diff = expected - actual;

      if (diff) {
        tot_err += 1;
        printf("Expected: %4ld  ", expected);
        printf("Actual: %4ld  ", actual);
        printf("Diff: %4ld at Index %12lu in Output %lu\r\n", diff, i, buf);
      }
    }
  }
  printf("Errors: %ld out of %ld \r\n", tot_err, tot);

  return tot_err;
}
