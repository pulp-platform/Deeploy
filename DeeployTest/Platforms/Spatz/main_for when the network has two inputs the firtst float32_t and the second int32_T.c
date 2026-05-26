
// main_for when the network has two inputs the firtst float32_t and the second int32_t
#include <stdint.h>
#include <benchmark.h>
#include "printf.h"

#include "Network.h"
#include "testinputs.h"
#include "testoutputs.h"

int main() {
  const unsigned int core_id = snrt_cluster_core_idx();
  unsigned int timer_start, timer_end, timer;

  if (core_id == 0) printf("[INFO] Running on %d cores\n", snrt_cluster_core_num());
  if (snrt_is_dm_core()){printf("[INFO] DM core is core number %d\n", core_id);}
  snrt_cluster_hw_barrier();

  // do it only with one of the two spatz cores
  if (snrt_is_dm_core()){
    timer_start = benchmark_get_cycle();

    printf("Initializing network...\r\n");
    InitNetwork(0, 1);

    for (uint32_t buf = 0; buf < DeeployNetwork_num_inputs; buf++) {                                                                                
      memcpy(DeeployNetwork_inputs[buf], testInputVector[buf], DeeployNetwork_inputs_bytes[buf]);
      // DeeployNetwork_inputs[buf] = (void *)testInputVector[buf]; TODO ???
    }   


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
    float32_t diff;
    float32_t expected, actual;

    for (uint32_t i = 0; i < DeeployNetwork_outputs_bytes[0] / sizeof(float32_t); i++) {
      expected = ((float32_t *)testOutputVector[0])[i];
      actual = ((float32_t *)DeeployNetwork_outputs[0])[i];
      diff = expected - actual;

      // MATTIA: if diff is a quiet nan 0x7FC00000 we want to error
      if ((diff < -1e-4f) || (diff > 1e-4f) || *(uint32_t*)&diff == 0x7FC00000) {
        tot_err += 1;
        printf("Expected: 0x%08x  Actual: 0x%08x  Diff: 0x%08x at Index %12u in Output %u\r\n", *(uint32_t*)&expected, *(uint32_t*)&actual, *(uint32_t*)&diff, i, 0);  
      }
    }

    int32_t diff_int;
    int32_t expected_int, actual_int;

    for (uint32_t i = 0; i < DeeployNetwork_outputs_bytes[1] / sizeof(int32_t); i++) {
      expected_int = ((int32_t *)testOutputVector[1])[i];
      actual_int = ((int32_t *)DeeployNetwork_outputs[1])[i];
      diff_int = expected_int - actual_int;

      if (diff_int != 0) {
        tot_err += 1;
        printf("Expected: %4d  ", expected_int);
        printf("Actual: %4d  ", actual_int);
        printf("Diff: %4d at Index %12u in Output %u\r\n", diff_int, i, 1);
      }
    }

      printf("first element of first output%08x\n", *(uint32_t*)&(((float32_t *)DeeployNetwork_outputs[0])[0]));

    printf("Errors: %d out of %d \r\n", tot_err, tot);
  }

  printf("core %d arrived at the end\r\n", core_id);
  snrt_cluster_hw_barrier();
  printf("We are after hw barrier\r\n");

  return 0;
}
