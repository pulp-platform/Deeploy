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

// RW: Remove MAINSTACKSIZE because gap9-sdk does not use it
#define SLAVESTACKSIZE 3800
#define WRITE_GPIO(x) pi_gpio_pin_write(89, x)

#ifdef POWER_MEASUREMENT
unsigned int GPIOs = 89;
#define WRITE_GPIO(x) pi_gpio_pin_write(GPIOs, x)
#endif

struct pi_device cluster_dev;
uint32_t total_cycles = 0;

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
    int nan_count = 0, inf_count = 0, first_bad = -1;
    float amin = 0.0f, amax = 0.0f;
    int seen_finite = 0;

    for (int i = 0; i < num_elements; i++) {
      float expected_val = expected[i];
      float actual_val = actual[i];
      float diff = expected_val - actual_val;
      int is_err = (diff < -1e-4) || (diff > 1e-4) || isnan(diff);

      if (isnan(actual_val)) {
        nan_count += 1;
        if (first_bad < 0)
          first_bad = i;
      } else if (isinf(actual_val)) {
        inf_count += 1;
        if (first_bad < 0)
          first_bad = i;
      } else {
        if (!seen_finite) {
          amin = amax = actual_val;
          seen_finite = 1;
        } else {
          if (actual_val < amin)
            amin = actual_val;
          if (actual_val > amax)
            amax = actual_val;
        }
      }

      if (is_err) {
        local_err_count += 1;
      }

      // Full per-element dump only for small outputs (e.g. final classifier)
      if (num_elements <= 32) {
        printf("Out %u[%4d] expected: %12.6f  actual: %12.6f  diff: %12.6f%s\r\n",
               output_buf_index, i, expected_val, actual_val, diff,
               is_err ? "  <-- ERR" : "");
      }
    }

    // Compact summary — works for any tensor size (use for NaN bisection)
    printf("[SUMMARY] Out %u: n=%d errs=%d nan=%d inf=%d first_bad=%d "
           "actual_min=%.6f actual_max=%.6f\r\n",
           output_buf_index, num_elements, local_err_count, nan_count,
           inf_count, first_bad, amin, amax);

    *err_count = local_err_count;
  }
}

void CL_CompareFloat(void *arg) {
  pi_cl_team_fork(NUM_CORES, CompareFloatOnCluster, arg);
}

void InitNetworkWrapper(void *args) {
  (void)args;
  InitNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

void RunNetworkWrapper(void *args) {
  (void)args;
  // Initialize performance counter in cluster context
  ResetTimer();
  StartTimer();
  RunNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
  total_cycles = getCycles();
  StopTimer();
}

int main(void) {

#ifdef POWER_MEASUREMENT
  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
#endif

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

  pi_cluster_task(&cluster_task, InitNetworkWrapper, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  printf("[Deeploy] nb_cores=%d slave_stack=%d stacks_total=%d\r\n",
         cluster_task.nb_cores, cluster_task.slave_stack_size,
         cluster_task.nb_cores * cluster_task.slave_stack_size);
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
   unsigned int GPIOs = 89;

  pi_pad_function_set(GPIOs, 1);
  pi_gpio_pin_configure(GPIOs, PI_GPIO_OUTPUT);
  pi_gpio_pin_write(GPIOs, 0);
  WRITE_GPIO(0);
  WRITE_GPIO(1);


  pi_cluster_task(&cluster_task, RunNetworkWrapper, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;

#ifdef POWER_MEASUREMENT
  WRITE_GPIO(1);
#endif

  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  WRITE_GPIO(0);

#ifdef POWER_MEASUREMENT
  WRITE_GPIO(0);
#endif

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

    if ((uint32_t)DeeployNetwork_outputs[buf] < 0x10000000) {
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

        // if (diff) {
        //   tot_err += 1;
        //   printf("Expected: %4d  ", expected);
        //   printf("Actual: %4d  ", actual);
        //   printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        // }
      }
    }
    if ((uint32_t)DeeployNetwork_outputs[buf] < 0x10000000) {
      pi_l2_free(compbuf, DeeployNetwork_outputs_bytes[buf]);
    }
  }

  printf("Runtime: %u cycles\r\n", total_cycles);
  printf("Errors: %u out of %u \r\n", tot_err, tot_tested);

  return 0;
}