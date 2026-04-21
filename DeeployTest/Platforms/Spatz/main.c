
#include <stdint.h>
#include <string.h>
#include <benchmark.h>
#include "printf.h"

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

int main() {
  const unsigned int core_id = snrt_cluster_core_idx();
  unsigned int timer_start, timer_end, timer;

  printf("Running on %d cores\r\n", snrt_cluster_core_num());
  if (snrt_is_dm_core()){printf("dm core is core number %d\r\n", core_id);}
  snrt_cluster_hw_barrier();

  // do it only with one of the two spatz cores
  if (snrt_is_dm_core()){
    timer_start = benchmark_get_cycle();

    printf("Initializing network...\r\n");
    InitNetwork(0, 1);

    // Copy inputs to allocated memory
    printf("Copying inputs to allocated memory...\r\n");
    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {
      snrt_dma_start_1d(DeeployNetwork_inputs[buf], testInputVector[buf], DeeployNetwork_inputs_bytes[buf]);
    }
    snrt_dma_wait_all();

    printf("Running network...\r\n");
  }

  snrt_cluster_hw_barrier();
  if (snrt_is_dm_core()){ timer_start = benchmark_get_cycle(); }
  RunNetwork(core_id, 2);
  
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
        if ((diff < -1e-4) || (diff > 1e-4)) {
          tot_err += 1;
          printf("Expected: %10.6f  ", (float)expected);
          printf("Actual: %10.6f  ", (float)actual);
          printf("Diff: %10.6f at Index %12u in Output %u\r\n", (float)diff, i,
                buf);
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