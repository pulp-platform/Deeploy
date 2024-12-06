/*
 * ----------------------------------------------------------------------
 *
 * File: deeploytest.c
 *
 * Last edited: 23.04.2024
 *
 * Copyright (C) 2024, ETH Zurich and University of Bologna.
 *
 * Author: Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
 *
 * ----------------------------------------------------------------------
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

#include "CycleCounter.h"
#include "Network.h"
#include "snrt.h"
#include "testinputs.h"
#include "testoutputs.h"

// #define NOPRINT
// #define NOTEST
// #define CI

int main(void) {

  uint32_t core_id = snrt_global_core_idx();
  uint32_t compute_core_id = snrt_global_compute_core_idx();

#ifdef BANSHEE_SIMULATION
  uint32_t const num_compute_cores = (NUM_CORES - 1);
#else
  uint32_t const num_compute_cores = snrt_global_compute_core_num();
#endif

  if (snrt_is_dm_core()) {
#ifndef CI
    printf("Network running on %d of %d compute cores (+%d DM cores) on %d "
           "clusters\r\n",
           num_compute_cores, snrt_global_compute_core_num(),
           snrt_cluster_num() * snrt_cluster_dm_core_num(), snrt_cluster_num());
#endif

#ifndef NOPRINT
    printf("Initializing...\r\n");
#endif
    InitNetwork(core_id, 1);

#ifndef CI
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
      printf("testInputVector%d @ %p\r\n", buf, testInputVector[buf]);
      printf("DeeployNetwork_input_%d @ %p and %u elements\r\n", buf,
             DeeployNetwork_inputs[buf], DeeployNetwork_inputs_bytes[buf]);
    }
    for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
      printf("testOutputVector%d @ %p\r\n", buf, testOutputVector[buf]);
      printf("DeeployNetwork_output_%d @ %p and %u elements\r\n", buf,
             DeeployNetwork_outputs[buf], DeeployNetwork_outputs_bytes[buf]);
    }

    printf("Initialized\r\n");
#endif

#ifndef NOPRINT
    printf("Copy inputs...\r\n");
#endif

    // WIESEP: Copy inputs to allocated memory
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
      snrt_dma_start_1d(DeeployNetwork_inputs[buf], testInputVector[buf],
                        DeeployNetwork_inputs_bytes[buf]);
    }
    snrt_dma_wait_all();

#ifndef CI
    printf("Input copied\r\n");
#endif
  }

#ifndef NOPRINT
  if (snrt_is_dm_core()) {
    printf("Running network...\r\n");
  }
#endif

  snrt_cluster_hw_barrier();

#ifndef BANSHEE_SIMULATION
  if (snrt_is_dm_core()) {
    ResetTimer();
    StartTimer();
  }
#endif // BANSHEE_SIMULATION

  RunNetwork(compute_core_id, num_compute_cores);

  uint32_t runtimeCycles = 0;
#ifndef BANSHEE_SIMULATION
  if (snrt_is_dm_core()) {
    runtimeCycles = getCycles();
    DUMP(runtimeCycles);
    StopTimer();
  }
#endif // BANSHEE_SIMULATION

  snrt_cluster_hw_barrier();

#ifndef CI
  if (snrt_is_dm_core()) {
    printf("Network done\r\n");
  }
#endif

  if (snrt_is_dm_core()) {
#ifndef CI
    printf("Output:\r\n");
#endif

#ifndef NOTEST
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
#ifndef CI
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
#endif
        }
      }
    }
    printf("Errors: %u out of %u \r\n", tot_err, tot);
#endif

#ifndef NOPRINT
    printf("Runtime: %u cycles\r\n", runtimeCycles);
#endif
  }

  snrt_cluster_hw_barrier();
  return 0;
}
