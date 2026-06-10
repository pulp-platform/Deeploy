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

// ---- One-shot L3/OctoSPI cluster-DMA self-test (diagnostic) ----
// Writes a known byte ramp into an L3 buffer, then reads it back three ways via
// the SAME pi_cl_ram_copy_2d API and verifies each against the ramp:
//   (a) one large contiguous block        (length == size)
//   (b) one small contiguous block        (200 B)
//   (c) strided gather 80x200 B @ stride 384  (the conv NCHW->NHWC transpose pattern)
// All three pass under GVSoC. On the board, whichever FAILs pinpoints the
// broken OctoSPI access pattern (contiguous vs small vs strided).
void L3SelfTest(void *arg) {
  (void)arg;
  if (pi_core_id() != 0)
    return;

  const uint32_t STRIDE = 384, LINE = 200, NLINES = 80;
  const uint32_t REGION = NLINES * STRIDE; // 30720 B written to L3
  const uint32_t SSZ = NLINES * LINE;      // 16000 B gathered by the strided read

  uint8_t *src = (uint8_t *)pi_l2_malloc(REGION);
  uint8_t *dst = (uint8_t *)pi_l2_malloc(REGION);
  uint8_t *sdst = (uint8_t *)pi_l2_malloc(SSZ);
  if (!src || !dst || !sdst) {
    printf("[L3TEST] L2 alloc failed\r\n");
    return;
  }
  for (uint32_t i = 0; i < REGION; i++)
    src[i] = (uint8_t)(i & 0xFF);

  void *l3 = cl_ram_malloc(REGION);
  cl_ram_write(l3, src, REGION); // contiguous write of the ramp into L3

  int bad;

  // (a) large contiguous read-back (single line: length == size)
  for (uint32_t i = 0; i < REGION; i++)
    dst[i] = 0xAA;
  {
    pi_cl_ram_req_t req = {0};
    pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t)l3, dst, REGION, REGION, REGION, 1, &req);
    pi_cl_ram_copy_wait(&req);
  }
  bad = -1;
  for (uint32_t i = 0; i < REGION; i++)
    if (dst[i] != (uint8_t)(i & 0xFF)) {
      bad = (int)i;
      break;
    }
  if (bad < 0)
    printf("[L3TEST] (a) contiguous %u B       : PASS\r\n", REGION);
  else
    printf("[L3TEST] (a) contiguous %u B       : FAIL @%d got %u exp %u\r\n", REGION, bad, dst[bad],
           (unsigned)(bad & 0xFF));

  // (b) small contiguous read (single 200 B line)
  for (uint32_t i = 0; i < LINE; i++)
    sdst[i] = 0xAA;
  {
    pi_cl_ram_req_t req = {0};
    pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t)l3, sdst, LINE, LINE, LINE, 1, &req);
    pi_cl_ram_copy_wait(&req);
  }
  bad = -1;
  for (uint32_t i = 0; i < LINE; i++)
    if (sdst[i] != (uint8_t)(i & 0xFF)) {
      bad = (int)i;
      break;
    }
  if (bad < 0)
    printf("[L3TEST] (b) small contiguous %u B  : PASS\r\n", LINE);
  else
    printf("[L3TEST] (b) small contiguous %u B  : FAIL @%d got %u exp %u\r\n", LINE, bad, sdst[bad],
           (unsigned)(bad & 0xFF));

  // (c) strided gather: NLINES lines of LINE bytes at STRIDE (conv-transpose pattern)
  for (uint32_t i = 0; i < SSZ; i++)
    sdst[i] = 0xAA;
  {
    pi_cl_ram_req_t req = {0};
    pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t)l3, sdst, SSZ, STRIDE, LINE, 1, &req);
    pi_cl_ram_copy_wait(&req);
  }
  bad = -1;
  for (uint32_t L = 0; L < NLINES && bad < 0; L++)
    for (uint32_t k = 0; k < LINE; k++)
      if (sdst[L * LINE + k] != (uint8_t)((L * STRIDE + k) & 0xFF)) {
        bad = (int)(L * LINE + k);
        break;
      }
  if (bad < 0)
    printf("[L3TEST] (c) strided %ux%u@%u    : PASS\r\n", NLINES, LINE, STRIDE);
  else {
    uint32_t L = (uint32_t)bad / LINE, k = (uint32_t)bad % LINE;
    printf("[L3TEST] (c) strided %ux%u@%u    : FAIL line %u byte %u got %u exp %u\r\n", NLINES, LINE, STRIDE, L, k,
           sdst[bad], (unsigned)((L * STRIDE + k) & 0xFF));
  }

  cl_ram_free(l3, REGION);
  pi_l2_free(src, REGION);
  pi_l2_free(dst, REGION);
  pi_l2_free(sdst, SSZ);
}

// Run the self-test via team fork (same path as CompareFloatOnCluster, whose
// core-0 printf reaches the UART; a bare cluster-task printf does not surface).
void L3SelfTestCl(void *arg) {
  pi_cl_team_fork(NUM_CORES, L3SelfTest, arg);
}

// ---- DEBUG: zero L1/L2 arenas to emulate GVSoC (remove after test) ----
// If the on-board FLT_MAX/NaN garbage is caused by the conv reading/scattering
// an uninitialized L1 tile (e.g. an under-written output tile), zeroing the
// scratch arenas before RunNetwork makes the board match GVSoC. Runs on the
// cluster so it can write cluster-L1; L1/L2 hold no persistent data between
// InitNetwork and RunNetwork (weights/inputs live in L3), so this is safe.
void ZeroArenas(void *arg) {
  (void)arg;
  if (pi_core_id() != 0)
    return;
  for (uint32_t i = 0; i < DeeployNetwork_MEMORYARENA_L1_len; i++)
    DeeployNetwork_MEMORYARENA_L1[i] = 0;
  for (uint32_t i = 0; i < DeeployNetwork_MEMORYARENA_L2_len; i++)
    DeeployNetwork_MEMORYARENA_L2[i] = 0;
  printf("[ZEROFILL] L1/L2 arenas zeroed (%u + %u B)\r\n",
         (unsigned)DeeployNetwork_MEMORYARENA_L1_len,
         (unsigned)DeeployNetwork_MEMORYARENA_L2_len);

  // Zero the L3 *scratch* region (everything after the loaded input): this is
  // where the conv reads its transposed input and where the network output
  // lands. The input itself (L3[0 .. input_0_len)) is preserved so the run is
  // valid. If the FLT_MAX/NaN garbage is uninitialized L3 read-before-write,
  // this makes the board match GVSoC.
  {
    uint8_t *l3_base = (uint8_t *)DeeployNetwork_MEMORYARENA_L3;
    const uint32_t start = DeeployNetwork_input_0_len * (uint32_t)sizeof(float32_t);
    const uint32_t total = DeeployNetwork_MEMORYARENA_L3_len;
    const uint32_t CHUNK = 4096;
    uint8_t *zbuf = (uint8_t *)pi_l2_malloc(CHUNK);
    for (uint32_t i = 0; i < CHUNK; i++)
      zbuf[i] = 0;
    for (uint32_t off = start; off < total; off += CHUNK) {
      uint32_t n = (total - off) < CHUNK ? (total - off) : CHUNK;
      cl_ram_write((void *)(l3_base + off), zbuf, n);
    }
    pi_l2_free(zbuf, CHUNK);
    printf("[ZEROFILL] L3 scratch zeroed [%u..%u)\r\n", (unsigned)start, (unsigned)total);
  }

  // DEBUG: read the loaded input back from L3 to verify the flash->L3 (readfs)
  // load path on silicon. GVSoC is correct with the identical binary+hex, so if
  // these bytes are wrong on the board the data load is the culprit; if they are
  // correct, the divergence is in the compute (FPU/kernels). Expected (0.hex):
  //   -0.45539  0.77912  0.17873  -0.73684  -0.98076  0.26855  0.18290  -2.02323
  {
    const uint32_t N = 8, BYTES = N * (uint32_t)sizeof(float32_t);
    float32_t *tmp = (float32_t *)pi_l2_malloc(BYTES);
    pi_cl_ram_req_t req = {0};
    pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t)DeeployNetwork_input_0, tmp, BYTES, BYTES, BYTES, 1, &req);
    pi_cl_ram_copy_wait(&req);
    printf("[DUMP] input_0[0..7]:");
    for (uint32_t i = 0; i < N; i++)
      printf(" %.5f", tmp[i]);
    printf("\r\n");
    pi_l2_free(tmp, BYTES);
  }
}

void ZeroArenasCl(void *arg) { pi_cl_team_fork(NUM_CORES, ZeroArenas, arg); }

// ---- DEBUG: scan the conv's pre-transpose output in L3 after the run ----
// output_pre_transposed lives at L3+0 (== DeeployNetwork_input_0's address) and
// is NOT overwritten by the final output transpose (which writes output_0 at
// L3+147456). So reading it post-run shows whether the conv ALREADY produced
// FLT_MAX/NaN (=> conv kernel is the culprit) or is clean (=> output transpose).
void DumpConvOut(void *arg) {
  (void)arg;
  if (pi_core_id() != 0)
    return;
  const uint32_t N = 36864, CH = 1024;
  float32_t *tmp = (float32_t *)pi_l2_malloc(CH * sizeof(float32_t));
  float mn = 1e30f, mx = -1e30f;
  uint32_t nanc = 0, infc = 0;
  for (uint32_t off = 0; off < N; off += CH) {
    uint32_t n = (N - off) < CH ? (N - off) : CH;
    pi_cl_ram_req_t req = {0};
    pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t)((char *)DeeployNetwork_input_0 + off * 4), tmp, n * 4, n * 4, n * 4, 1, &req);
    pi_cl_ram_copy_wait(&req);
    for (uint32_t i = 0; i < n; i++) {
      float v = tmp[i];
      if (isnan(v))
        nanc++;
      else if (isinf(v))
        infc++;
      else {
        if (v < mn)
          mn = v;
        if (v > mx)
          mx = v;
      }
    }
  }
  printf("[CONVOUT] pre-transpose @L3+0: nan=%u inf=%u min=%f max=%f first:", (unsigned)nanc, (unsigned)infc, mn, mx);
  pi_cl_ram_req_t req = {0};
  pi_cl_ram_copy_2d(get_ram_ptr(), (uint32_t)DeeployNetwork_input_0, tmp, 32, 32, 32, 1, &req);
  pi_cl_ram_copy_wait(&req);
  for (uint32_t i = 0; i < 8; i++)
    printf(" %.5f", tmp[i]);
  printf("\r\n");
  pi_l2_free(tmp, CH * sizeof(float32_t));
}

void DumpConvOutCl(void *arg) { pi_cl_team_fork(NUM_CORES, DumpConvOut, arg); }

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

  // --- L3/OctoSPI DMA diagnostic (remove after debugging) ---
  {
    struct pi_cluster_task l3test_task;
    pi_cluster_task(&l3test_task, L3SelfTestCl, NULL);
    l3test_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &l3test_task);
  }

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


  // ---- DEBUG: zero scratch arenas before the run (remove after test) ----
  {
    struct pi_cluster_task zero_task;
    pi_cluster_task(&zero_task, ZeroArenasCl, NULL);
    zero_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &zero_task);
  }

  pi_cluster_task(&cluster_task, RunNetworkWrapper, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;

#ifdef POWER_MEASUREMENT
  WRITE_GPIO(1);
#endif

  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  WRITE_GPIO(0);

  // ---- DEBUG: scan conv pre-transpose output after the run (remove later) ----
  {
    struct pi_cluster_task dump_task;
    pi_cluster_task(&dump_task, DumpConvOutCl, NULL);
    dump_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &dump_task);
  }

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