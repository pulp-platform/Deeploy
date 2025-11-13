/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>

#include "CycleCounter.h"
#include "Network.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

#define MAINSTACKSIZE 8000
#define SLAVESTACKSIZE 3800

struct pi_device cluster_dev;

typedef struct {
  void *expected;
  void *actual;
  int num_elements;
  int output_buf_index;
  int *err_count;
} FloatCompareArgs;

void CompareFloatOnCluster(void *args) {

  if (pi_core_id() == 0) {
    FloatCompareArgs *compare_args = (FloatCompareArgs *)args;
    float *expected = (float *)compare_args->expected;
    float *actual = (float *)compare_args->actual;
    int num_elements = compare_args->num_elements;
    int output_buf_index = compare_args->output_buf_index;
    int *err_count = compare_args->err_count;

    int local_err_count = 0;

    for (int i = 0; i < num_elements; i++) {
      float expected_val = expected[i];
      float actual_val = actual[i];
      float diff = expected_val - actual_val;

      if ((diff < -1e-4) || (diff > 1e-4) || isnan(diff)) {
        local_err_count += 1;

        printf("Expected: %10.6f  ", expected_val);
        printf("Actual: %10.6f  ", actual_val);
        printf("Diff: %10.6f at Index %12u in Output %u\r\n", diff, i,
               output_buf_index);
      }
    }

    *err_count = local_err_count;
  }
}

void CL_CompareFloat(void *arg) {
  pi_cl_team_fork(NUM_CORES, CompareFloatOnCluster, arg);
}

void PE_RunNetwork(void *arg) {
#ifndef CI
  uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
  printf("[%d %d] Run Network!\n", cluster_id, core_id);
#endif
  RunNetwork(pi_core_id(), NUM_CORES);
}

void CL_RunNetwork(void *arg) {
  pi_cl_team_fork(NUM_CORES, PE_RunNetwork, NULL);
}

void PE_InitNetwork(void *arg) {
#ifndef CI
  uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
  printf("[%d %d] Init Network!\n", cluster_id, core_id);
#endif

  InitNetwork(pi_core_id(), NUM_CORES);
}

int main(void) {
#ifndef CI
  uint32_t core_id = pi_core_id(), cluster_id = pi_cluster_id();
  printf("[%d %d] Hello World!\n", cluster_id, core_id);
#endif
  struct pi_cluster_conf conf;

  pi_cluster_conf_init(&conf);
  conf.id = 0;
  pi_open_from_conf(&cluster_dev, &conf);
  if (pi_cluster_open(&cluster_dev))
    return -1;

  mem_init();
#ifndef NOFLASH
  open_fs();
#endif

  printf("Intializing\r\n");

  struct pi_cluster_task cluster_task;

  pi_cluster_task(&cluster_task, PE_InitNetwork, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

#ifndef CI
  printf("Initialized\r\n");
#endif
  for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
    if ((uint32_t)DeeployNetwork_inputs[buf] >= 0x10000000) {
      memcpy(DeeployNetwork_inputs[buf], testInputVector[buf],
             DeeployNetwork_inputs_bytes[buf]);
    }
  }

#ifndef CI
  printf("Input copied\r\n");
#endif

  pi_cluster_task(&cluster_task, CL_RunNetwork, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  ResetTimer();
  StartTimer();
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  StopTimer();

#ifndef CI
  printf("Output:\r\n");
#endif

  uint32_t tot_err, tot_tested;
  tot_err = 0;
  tot_tested = 0;
  void *compbuf;
  FloatCompareArgs float_compare_args;
  uint32_t float_error_count = 0;

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    tot_tested += DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE);

    if ((uint32_t)DeeployNetwork_outputs[buf] < 0x1000000) {
      compbuf = pi_l2_malloc(DeeployNetwork_outputs_bytes[buf]);
      ram_read(compbuf, DeeployNetwork_outputs[buf],
               DeeployNetwork_outputs_bytes[buf]);
    } else {
      compbuf = DeeployNetwork_outputs[buf];
    }

    if (ISOUTPUTFLOAT) {
      float_error_count = 0;
      float_compare_args.expected = testOutputVector[buf];
      float_compare_args.actual = compbuf;
      float_compare_args.num_elements =
          DeeployNetwork_outputs_bytes[buf] / sizeof(float);
      float_compare_args.output_buf_index = buf;
      float_compare_args.err_count = (int *)&float_error_count;

      pi_cluster_task(&cluster_task, CL_CompareFloat, &float_compare_args);
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      tot_err += float_error_count;
    } else {

      for (uint32_t i = 0;
           i < DeeployNetwork_outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
        OUTPUTTYPE expected = ((OUTPUTTYPE *)testOutputVector[buf])[i];
        OUTPUTTYPE actual = ((OUTPUTTYPE *)compbuf)[i];
        OUTPUTTYPE diff = expected - actual;

        if (diff) {
          tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
      }
    }
    if ((uint32_t)DeeployNetwork_outputs[buf] < 0x1000000) {
      pi_l2_free(compbuf, DeeployNetwork_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", getCycles());
  printf("Errors: %u out of %u \r\n", tot_err, tot_tested);

  return 0;
}