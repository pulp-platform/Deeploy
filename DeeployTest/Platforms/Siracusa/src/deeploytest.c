/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
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
#define FLOAT_ABS_TOL 1e-4f
#define FLOAT_REL_TOL 1e-5f
// Outputs that live in L3 are read back and compared in chunks of at most this
// many bytes: the expected output already occupies L2, so a second L2 buffer
// for the whole output may not fit.
#define L3_COMPARE_CHUNK_BYTES 65536

struct pi_device cluster_dev;

typedef struct {
  void *expected;
  void *actual;
  uint32_t num_elements;
  uint32_t index_offset;
  uint32_t output_buf_index;
  uint32_t *err_count;
} FloatCompareArgs;

void CompareFloatOnCluster(void *args) {

  if (pi_core_id() == 0) {
    FloatCompareArgs *compare_args = (FloatCompareArgs *)args;
    float *expected = (float *)compare_args->expected;
    float *actual = (float *)compare_args->actual;
    uint32_t num_elements = compare_args->num_elements;
    uint32_t index_offset = compare_args->index_offset;
    uint32_t output_buf_index = compare_args->output_buf_index;
    uint32_t *err_count = compare_args->err_count;

    uint32_t local_err_count = 0;

    for (uint32_t i = 0; i < num_elements; i++) {
      float expected_val = expected[i];
      float actual_val = actual[i];
      float diff = expected_val - actual_val;
      float abs_diff = fabsf(diff);
      float scale = fabsf(expected_val);
      float abs_actual = fabsf(actual_val);
      if (abs_actual > scale) {
        scale = abs_actual;
      }
      float tolerance = FLOAT_ABS_TOL + FLOAT_REL_TOL * scale;

      // Non-finite values are errors: inf > inf is false and would hide a
      // mismatch.
      if (!isfinite(expected_val) || !isfinite(actual_val) ||
          (abs_diff > tolerance)) {
        local_err_count += 1;

        printf("Expected: %10.6f  ", expected_val);
        printf("Actual: %10.6f  ", actual_val);
        printf("Diff: %10.6f at Index %12u in Output %u\r\n", diff,
               index_offset + i, output_buf_index);
      }
    }

    *err_count = local_err_count;
  }
}

// Compares num_elements outputs starting at element index_offset of output
// buffer buf. Returns the number of mismatches.
static uint32_t CompareOutputChunk(void *expected, void *actual,
                                   uint32_t num_elements, uint32_t index_offset,
                                   uint32_t buf) {
#if ISOUTPUTFLOAT == 1
  uint32_t float_error_count = 0;
  FloatCompareArgs float_compare_args;
  struct pi_cluster_task cluster_task;

  float_compare_args.expected = expected;
  float_compare_args.actual = actual;
  float_compare_args.num_elements = num_elements;
  float_compare_args.index_offset = index_offset;
  float_compare_args.output_buf_index = buf;
  float_compare_args.err_count = &float_error_count;

  pi_cluster_task(&cluster_task, CompareFloatOnCluster, &float_compare_args);
  cluster_task.stack_size = MAINSTACKSIZE;
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  return float_error_count;
#else
  uint32_t err = 0;
  for (uint32_t i = 0; i < num_elements; i++) {
    OUTPUTTYPE expected_val = ((OUTPUTTYPE *)expected)[i];
    OUTPUTTYPE actual_val = ((OUTPUTTYPE *)actual)[i];
    int32_t error = expected_val - actual_val;
    OUTPUTTYPE diff = (OUTPUTTYPE)(error < 0 ? -error : error);

    if (diff) {
      err += 1;
      printf("Expected: %4d  ", expected_val);
      printf("Actual: %4d  ", actual_val);
      printf("Diff: %4d at Index %12u in Output %u\r\n", diff, index_offset + i,
             buf);
    }
  }
  return err;
#endif
}

int main(void) {
#ifndef CI
  printf("HELLO WORLD:\r\n");
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

  pi_cluster_task(&cluster_task, InitNetwork, NULL);
  cluster_task.stack_size = MAINSTACKSIZE;
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

  pi_cluster_task(&cluster_task, RunNetwork, NULL);
  cluster_task.stack_size = MAINSTACKSIZE;
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

  for (uint32_t buf = 0; buf < DeeployNetwork_num_outputs; buf++) {
    uint32_t bytes = DeeployNetwork_outputs_bytes[buf];
    uint32_t num_elements = bytes / sizeof(OUTPUTTYPE);
    tot_tested += num_elements;

    if ((uint32_t)DeeployNetwork_outputs[buf] >= 0x1000000) {
      tot_err +=
          CompareOutputChunk(testOutputVector[buf], DeeployNetwork_outputs[buf],
                             num_elements, 0, buf);
      continue;
    }

    // Output in L3: stream it through an L2 buffer.
    uint32_t chunk_bytes = bytes < L3_COMPARE_CHUNK_BYTES
                               ? bytes
                               : (L3_COMPARE_CHUNK_BYTES / sizeof(OUTPUTTYPE)) *
                                     sizeof(OUTPUTTYPE);
    void *compbuf = pi_l2_malloc((int)chunk_bytes);
    if (compbuf == NULL) {
      printf("Failed to allocate %u bytes of L2 to read back output %u\r\n",
             chunk_bytes, buf);
      return -1;
    }

    for (uint32_t offset = 0; offset < bytes; offset += chunk_bytes) {
      uint32_t n = bytes - offset < chunk_bytes ? bytes - offset : chunk_bytes;
      ram_read(compbuf, (char *)DeeployNetwork_outputs[buf] + offset, n);
      tot_err += CompareOutputChunk((char *)testOutputVector[buf] + offset,
                                    compbuf, n / sizeof(OUTPUTTYPE),
                                    offset / sizeof(OUTPUTTYPE), buf);
    }

    pi_l2_free(compbuf, (int)chunk_bytes);
  }

  printf("Runtime: %u cycles\r\n", getCycles());
  printf("Errors: %u out of %u \r\n", tot_err, tot_tested);

  return (int)tot_err;
}
