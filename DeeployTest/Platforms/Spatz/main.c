
#include <stdint.h>
#include <stddef.h>
#include <benchmark.h>
#include "printf.h"

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

#ifndef DEEPLOY_ZERO_COPY_TEST_INPUTS
#define DEEPLOY_ZERO_COPY_TEST_INPUTS 1
#endif

// Optional: some generated networks provide this helper to avoid copying
// test inputs into Deeploy-owned buffers.
#ifndef DEEPLOYNETWORK_HAS_BIND_EXTERNAL_INPUTS
void DeeployNetwork_BindExternalInputs(void **external_inputs) __attribute__((weak));
#endif


int main() {
  const unsigned int core_id = snrt_cluster_core_idx();
  unsigned int timer_start, timer_end, timer;

  if (core_id == 0) printf("[INFO] Running on %d cores\n", snrt_cluster_core_num());
  if (snrt_is_dm_core()){printf("[INFO] DM core is core number %d\n", core_id);}
  snrt_cluster_hw_barrier();

  // do it only with one of the two spatz cores
  if (snrt_is_dm_core()){
    printf("Initializing network...\r\n");
    InitNetwork(0, 1);

    // printf("Copying inputs to l3 buffer...\r\n");
#if DEEPLOY_ZERO_COPY_TEST_INPUTS
    if (DeeployNetwork_BindExternalInputs) {
      DeeployNetwork_BindExternalInputs(testInputVector);
    } else {
      for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
        memcpy(DeeployNetwork_inputs[buf], testInputVector[buf], DeeployNetwork_inputs_bytes[buf]);
      }
    }
#else
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
      memcpy(DeeployNetwork_inputs[buf], testInputVector[buf], DeeployNetwork_inputs_bytes[buf]);
    }
#endif

    printf("Running network...\r\n");
  }
  snrt_cluster_hw_barrier();

  if (snrt_is_dm_core()){ timer_start = benchmark_get_cycle(); }
  RunNetwork(core_id, 2);

  snrt_cluster_hw_barrier();
  
  if (snrt_is_dm_core()){
    timer_end = benchmark_get_cycle();
    timer = timer_end - timer_start;

    printf("Network ran in %d cycles.\r\nChecking Outputs...\r\n", timer);
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

#if ISOUTPUTFLOAT == 1
        // RUNWANG: Allow margin of error for float32_t
        // MATTIA: if diff is a quiet nan 0x7FC00000 we want to error
        if ((diff < -1e-4f) || (diff > 1e-4f) || *(uint32_t*)&diff == 0x7FC00000) {
          tot_err += 1;
          // printf("Expected: %f  Actual: %f  Diff: %f at Index %12u in Output %u\r\n", expected, actual, diff, i, buf);  
          printf("Expected: 0x%08x  Actual: 0x%08x  Diff: 0x%08x at Index %12u in Output %u\r\n", *(uint32_t*)&expected, *(uint32_t*)&actual, *(uint32_t*)&diff, i, buf);  
        }
#else
        // RUNWANG: No margin for integer comparison
        if (diff != 0) {
          tot_err += 1;
          printf("Expected: %4d  ", expected);
          printf("Actual: %4d  ", actual);
          printf("Diff: %4d at Index %12u in Output %u\r\n", diff, i, buf);
        }
#endif
      }
    }

    printf("Errors: %d out of %d \r\n", tot_err, tot);
  }

  printf("core %d arrived at the end\r\n", core_id);
  snrt_cluster_hw_barrier();
  printf("We are after hw barrier\r\n");

  return 0;
}
