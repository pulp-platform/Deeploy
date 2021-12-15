/* =====================================================================
 * Title:        deeploytest.c
 * Description:
 *
 * $Date:        26.12.2021
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

#include "CycleCounter.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

struct pi_device cluster_dev;

void main(void) {
#ifndef CI
  printf("HELLO WORLD:\r\n");
#endif

  struct pi_cluster_conf conf;

  pi_cluster_conf_init(&conf);
  conf.id = 0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return;

  struct pi_cluster_task cluster_task_mem_init;

  pi_cluster_task(&cluster_task_mem_init, mem_init, NULL);
  cluster_task_mem_init.stack_size = 5000;
  cluster_task_mem_init.slave_stack_size = 3800;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task_mem_init);

  struct pi_cluster_task cluster_task;

  pi_cluster_task(&cluster_task, InitNetwork, NULL);
  cluster_task.stack_size = 5000;
  cluster_task.slave_stack_size = 3800;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#ifndef CI
  printf("Initialized\r\n");
#endif
  for (int buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
           DeeployNetwork_inputs_bytes[buf]);
  }
#ifndef CI
  printf("Input copied\r\n");
#endif
  ResetTimer();
  StartTimer();

  // RunNetwork(0, 1);
  pi_cluster_task(&cluster_task, RunNetwork, NULL);
  cluster_task.stack_size = 5000;
  cluster_task.slave_stack_size = 3800;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
#ifndef CI
  printf("Run\r\n");
#endif
  StopTimer();
#ifndef CI
  printf("Output:\r\n");
#endif
  int32_t diff, tot_err;
  tot_err = 0;
  for (int buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    for (int i = 0; i < DeeployNetwork_outputs_bytes[buf]; i++) {
      diff = ((char *)testOutputVector[buf])[i] -
             ((char *)DeeployNetwork_outputs[buf])[i];
      if (diff) {
        tot_err += 1;
#ifndef CI
        printf("Expected: %i\t\t", ((int8_t *)testOutputVector[buf])[i]);
        printf("Actual: %i \t\t", ((int8_t *)DeeployNetwork_outputs[buf])[i]);
#endif
#ifndef CI
        printf("Diff: %i at Index %u \r\n", diff, i);
#endif
      } else {
        /* #ifndef CI */
        /*       printf("\r\n"); */
        /* #endif */
      }
    }
  }
  printf("Runtime: %u cycles\r\n", getCycles());
  printf("Errors: %u out of %u \r\n", tot_err, DeeployNetwork_output_0_len);
}
