/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
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
