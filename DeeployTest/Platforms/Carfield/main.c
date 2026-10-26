/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "Network.h"
#include "Utils.h"
#include "pulp.h"
#include "testinputs.h"
#include "testoutputs.h"

int main() {

  uint32_t core_id = get_core_id();
  uint32_t numThreads = get_core_num();

  BEGIN_SINGLE_CORE;
  // Initializing the network and allocating memory for inputs and outputs.
  InitNetwork(0, 1);

  // Copying the test input vector to the network input buffers.
  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
           DeeployNetwork_inputs_bytes[buf]);
  }
  END_SINGLE_CORE;
  // Synchronizing all cores before running the network.
  synch_barrier();

  // Running the network.
  RunNetwork(core_id, numThreads);

  // Let core 0 print the results, while the other cores return.
  if (core_id != 0) {
    return 0;
  }

  // Printing how much of the DDR arena is still free; `bytes_left` is declared
  // in Utils.h. The amount consumed is the arena size minus this.
  printf("DDR arena: %u bytes left\n\r", (unsigned)bytes_left);

  // With buffer reuse, bytes_left only says what is still live at the end.
  // bytes_peak is the high-water mark: the smallest arena this network runs in.
  printf("DDR arena: %u bytes peak\n\r", (unsigned)bytes_peak);

  // Printing the address of the output buffer, so the host can read it back.
  printf("Output: 0x%08x\n\r", (unsigned)DeeployNetwork_outputs[0]);

  // Comparing the network output buffers with the expected output vector.
  int32_t tot_err = 0;
  uint32_t tot = 0;
  const int32_t max_reported = 32;
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
        if (tot_err <= max_reported) {
          printf("Expected: %10.6f  ", (float)expected);
          printf("Actual: %10.6f  ", (float)actual);
          printf("Diff: %10.6f at Index %12u in Output %u\r\n", (float)diff, i,
                 buf);
        } else if (tot_err == max_reported + 1) {
          printf("... further mismatches not printed\r\n");
        }
      }
#else
      // RUNWANG: No margin for integer comparison
      if (diff != 0) {
        tot_err += 1;
        if (tot_err <= max_reported) {
          printf("Expected: %4d  ", (int32_t)expected);
          printf("Actual: %4d  ", (int32_t)actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", (int32_t)diff, i,
                 buf);
        } else if (tot_err == max_reported + 1) {
          printf("... further mismatches not printed\r\n");
        }
      }
#endif
    }
  }

  printf("Errors: %u out of %u \r\n", tot_err, tot);

  return tot_err;
}
