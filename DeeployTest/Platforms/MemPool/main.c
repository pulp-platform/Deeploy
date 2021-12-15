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
#include "dma.h"
#include "encoding.h"
#include "printf.h"
#include "runtime.h"
#include "synchronization.h"
#include "testinputs.h"
#include "testoutputs.h"

#ifndef BANSHEE_SIMULATION
dump(timer_cycle, 0);
dump(timer_instr, 1);
dump(expected, 2);
dump(actual, 3);
dump(diff, 4);
dump(info, 5);
dump(input0, 6);
dump(input1, 7);
#endif

#ifndef NUM_THREADS
#define NUM_THREADS 1
#endif

static int8_t inference_done __attribute__((section(".l1"))) = 0;

int main() {
  uint32_t const core_id = mempool_get_core_id();

  mempool_timer_t instr_init, instr_end;
  mempool_timer_t timer_init, timer_end;
#ifdef BANSHEE_SIMULATION
  uint32_t const num_cores = NUM_THREADS;
#else
  uint32_t const num_cores = mempool_get_core_count();
#endif

  mempool_init(core_id);

  // Initialize synchronization variables
  mempool_barrier_init(core_id, num_cores);

#ifdef BANSHEE_SIMULATION
  if (core_id == num_cores - 1) {
    printf("Network running on %ld of %ld cores\r\n", num_cores,
           mempool_get_core_count());
  }
#endif

  // Wait until initialization is done
  mempool_barrier(num_cores);

#ifdef BANSHEE_SIMULATION
  if (core_id == 0) {
    printf("Init network...\r\n");
  }
#endif

  if (core_id == 0) {
    InitNetwork(core_id, num_cores);
  }

  // Wait until initialization is done
  mempool_barrier(num_cores);

  if (core_id == 0) {
#if BANSHEE_SIMULATION
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
      memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
             DeeployNetwork_inputs_bytes[buf]);
    }
#else
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
      dma_memcpy_nonblocking(DeeployNetwork_inputs[buf], testInputVector[buf],
                             DeeployNetwork_inputs_bytes[buf]);
    }
    do {
      mempool_wait(16);
    } while (!dma_done());
#endif
  }

  // Wait until initialization is done
  mempool_barrier(num_cores);

#ifdef BANSHEE_SIMULATION
  if (core_id == 0) {
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
      printf("testInputVector%d @ %p\r\n", buf, testInputVector[buf]);
      printf("DeeployNetwork_input_%d @ %p and %lu elements\r\n", buf,
             DeeployNetwork_inputs[buf], DeeployNetwork_inputs_bytes[buf]);
    }
    for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
      printf("testInputVector%d @ %p\r\n", buf, testOutputVector[buf]);
      printf("DeeployNetwork_output_%d @ %p and %lu elements\r\n", buf,
             DeeployNetwork_outputs[buf], DeeployNetwork_outputs_bytes[buf]);
    }
    printf("Running network...\r\n");
  }
#endif

  instr_init = read_csr(minstret);
  timer_init = read_csr(mcycle);
  if (core_id < NUM_THREADS)
    RunNetwork(core_id, NUM_THREADS);
  timer_end = read_csr(mcycle);
  instr_end = read_csr(minstret);

  if (core_id == 0) {
    inference_done = 1;
    mempool_wait(64);
    wake_up_all();
  } else {
    while (inference_done == 0) {
      mempool_wfi();
    }
  }

  // Wait until all cores are done
  mempool_barrier(num_cores);

  int32_t tot_err = 0;
  int32_t diff;
  int32_t expected, actual;

#ifdef BANSHEE_SIMULATION
  uint32_t tot = 0;
  // Sequential part executed by all cores
  if (core_id != 0) {
    mempool_wfi();
  }
  printf("RunNetwork(%3ld, %3ld) Runtime: %6ld cycles, %6ld instr\r\n", core_id,
         num_cores, timer_end - timer_init, instr_end - instr_init - 2);
  wake_up(core_id + 1);

  // Wait until all cores are done
  mempool_barrier(num_cores);

  if (core_id == 0) {
    printf("Done. Checking outputs...\r\n");

    for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
      tot += DeeployNetwork_outputs_bytes[buf];
      for (uint32_t i = 0; i < DeeployNetwork_outputs_bytes[buf]; i++) {
        expected = ((char *)testOutputVector[buf])[i];
        actual = ((char *)DeeployNetwork_outputs[buf])[i];
        diff = expected - actual;

        if (diff) {
          tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    printf("Errors: %ld out of %lu \r\n", tot_err, tot);
  }
#else
  if (core_id != 0) {
    mempool_wfi();
  }
  dump_timer_cycle(timer_end - timer_init);
  dump_timer_instr(instr_end - instr_init - 2);
  // printf("RunNetwork(%3ld, %3ld) Runtime: %6ld cycles, %6ld instr\r\n",
  // core_id, num_cores, timer_end - timer_init,
  //       instr_end - instr_init - 2);
  wake_up(core_id + 1);

  // Wait until all cores are done
  mempool_barrier(num_cores);

  if (core_id == 0) {
    for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
      for (uint32_t i = 0; i < DeeployNetwork_outputs_bytes[buf]; i++) {
        expected = ((char *)testOutputVector[buf])[i];
        actual = ((char *)DeeployNetwork_outputs[buf])[i];

        diff = expected - actual;

        if (diff) {
          dump_expected((uint32_t)expected);
          dump_actual((uint32_t)actual);
          dump_diff((uint32_t)diff);
          tot_err += 1;
        }
      }
    }

    dump_info((uint32_t)tot_err);
  }
#endif

  // Wait until all cores have finished
  mempool_barrier(num_cores);

  return tot_err;
}
