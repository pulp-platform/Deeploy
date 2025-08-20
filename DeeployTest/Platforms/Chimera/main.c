/*
 * Copyright 2025 ETH Zurich.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Victor Jung <jungvi@iis.ee.ethz.ch>
 */

#include <stdio.h>

#include "uart.h"

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

      if (diff != 0) {
        tot_err += 1;
        printf("Expected: %4d  ", expected);
        printf("Actual: %4d  ", actual);
        printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
      }
    }
  }

  printf("Errors: %d out of %d \r\n", tot_err, tot);

  return 0;
}