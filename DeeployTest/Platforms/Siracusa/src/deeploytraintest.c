/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "CycleCounter.h"
#include "OptimizerNetwork.h"
#include "TrainingNetwork.h"
#include "dory_mem.h"
#include "pmsis.h"
#include "testinputs.h"
#include "testoutputs.h"

/* Compile-time defaults — override via CMake target_compile_definitions */

#ifndef N_TRAIN_STEPS
#define N_TRAIN_STEPS 1
#endif

#ifndef N_ACCUM_STEPS
#define N_ACCUM_STEPS 1
#endif

#ifndef TRAINING_NUM_DATA_INPUTS
#define TRAINING_NUM_DATA_INPUTS 2
#endif

#define MAINSTACKSIZE  8000
#define SLAVESTACKSIZE 3800

/* Cluster device */

struct pi_device cluster_dev;

/* Loss storage (one value per forward pass) */

#define TOTAL_FWD_PASSES (N_TRAIN_STEPS * N_ACCUM_STEPS)
static float stored_losses[TOTAL_FWD_PASSES];

/* Optimizer buffer helpers */

#ifdef OPTIMIZER_ADAM
  #define OPT_SHARED_INPUTS 2u
  #define OPT_PER_WEIGHT    4u
  #define OPT_R_IDX         0u
  #define OPT_T_IDX         1u
  #define OPT_W_IDX(wi)     (OPT_SHARED_INPUTS + OPT_PER_WEIGHT * (wi))
  #define OPT_G_IDX(wi)     (OPT_SHARED_INPUTS + OPT_PER_WEIGHT * (wi) + 1u)
  #define OPT_V_IDX(wi)     (OPT_SHARED_INPUTS + OPT_PER_WEIGHT * (wi) + 2u)
  #define OPT_H_IDX(wi)     (OPT_SHARED_INPUTS + OPT_PER_WEIGHT * (wi) + 3u)
#else
  #define OPT_SHARED_INPUTS 0u
  #define OPT_PER_WEIGHT    2u
  #define OPT_W_IDX(wi)     (2u * (wi))
  #define OPT_G_IDX(wi)     (2u * (wi) + 1u)
#endif

/* Adam state — step counter (incremented each optimizer step) */
#ifdef OPTIMIZER_ADAM
static int32_t adam_step_counter = 0;
#endif

static void init_optimizer_state(void) {
#ifdef OPTIMIZER_ADAM
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t v_idx = OPT_V_IDX(wi);
    uint32_t h_idx = OPT_H_IDX(wi);
    if ((uint32_t)DeeployOptNetwork_inputs[v_idx] >= 0x10000000u)
      memset(DeeployOptNetwork_inputs[v_idx], 0, DeeployOptNetwork_inputs_bytes[v_idx]);
    if ((uint32_t)DeeployOptNetwork_inputs[h_idx] >= 0x10000000u)
      memset(DeeployOptNetwork_inputs[h_idx], 0, DeeployOptNetwork_inputs_bytes[h_idx]);
  }
  if ((uint32_t)DeeployOptNetwork_inputs[OPT_R_IDX] >= 0x10000000u) {
    float lr = TRAINING_LEARNING_RATE;
    memcpy(DeeployOptNetwork_inputs[OPT_R_IDX], &lr, sizeof(float));
  }
  adam_step_counter = 0;
  if ((uint32_t)DeeployOptNetwork_inputs[OPT_T_IDX] >= 0x10000000u) {
    memcpy(DeeployOptNetwork_inputs[OPT_T_IDX], &adam_step_counter, sizeof(int32_t));
  }
#endif
#endif
}

/* run_optimizer_step — returns kernel cycles via *kernel_cycles_out */
static void run_optimizer_step(uint32_t *kernel_cycles_out) {
  *kernel_cycles_out = 0;
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)

#ifdef OPTIMIZER_ADAM
  adam_step_counter++;
  if ((uint32_t)DeeployOptNetwork_inputs[OPT_T_IDX] >= 0x10000000u) {
    memcpy(DeeployOptNetwork_inputs[OPT_T_IDX], &adam_step_counter, sizeof(int32_t));
  }
#endif
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t train_g_idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + wi;
    uint32_t opt_w_in    = OPT_W_IDX(wi);
    uint32_t opt_g_in    = OPT_G_IDX(wi);

    if ((uint32_t)DeeployNetwork_inputs[train_w_idx] >= 0x10000000u &&
        (uint32_t)DeeployOptNetwork_inputs[opt_w_in] >= 0x10000000u) {
      memcpy(DeeployOptNetwork_inputs[opt_w_in],
             DeeployNetwork_inputs[train_w_idx],
             DeeployNetwork_inputs_bytes[train_w_idx]);
    }
    if ((uint32_t)DeeployNetwork_inputs[train_g_idx] >= 0x10000000u &&
        (uint32_t)DeeployOptNetwork_inputs[opt_g_in] >= 0x10000000u) {
      memcpy(DeeployOptNetwork_inputs[opt_g_in],
             DeeployNetwork_inputs[train_g_idx],
             DeeployNetwork_inputs_bytes[train_g_idx]);
    }
  }
  ResetTimer();
  StartTimer();

  struct pi_cluster_task opt_task;
  pi_cluster_task(&opt_task, RunOptimizerNetwork, NULL);
  opt_task.stack_size       = MAINSTACKSIZE;
  opt_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &opt_task);

  StopTimer();
  *kernel_cycles_out = getCycles();
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;

#ifdef OPTIMIZER_ADAM
    uint32_t opt_v_out = 3u * wi;
    uint32_t opt_h_out = 3u * wi + 1u;
    uint32_t opt_w_out = 3u * wi + 2u;
    if ((uint32_t)DeeployOptNetwork_outputs[opt_w_out] >= 0x10000000u &&
        (uint32_t)DeeployNetwork_inputs[train_w_idx] >= 0x10000000u) {
      memcpy(DeeployNetwork_inputs[train_w_idx],
             DeeployOptNetwork_outputs[opt_w_out],
             DeeployNetwork_inputs_bytes[train_w_idx]);
    }
    uint32_t opt_v_in = OPT_V_IDX(wi);
    if ((uint32_t)DeeployOptNetwork_outputs[opt_v_out] >= 0x10000000u &&
        (uint32_t)DeeployOptNetwork_inputs[opt_v_in] >= 0x10000000u) {
      memcpy(DeeployOptNetwork_inputs[opt_v_in],
             DeeployOptNetwork_outputs[opt_v_out],
             DeeployOptNetwork_inputs_bytes[opt_v_in]);
    }
    uint32_t opt_h_in = OPT_H_IDX(wi);
    if ((uint32_t)DeeployOptNetwork_outputs[opt_h_out] >= 0x10000000u &&
        (uint32_t)DeeployOptNetwork_inputs[opt_h_in] >= 0x10000000u) {
      memcpy(DeeployOptNetwork_inputs[opt_h_in],
             DeeployOptNetwork_outputs[opt_h_out],
             DeeployOptNetwork_inputs_bytes[opt_h_in]);
    }
#else
    uint32_t opt_w_out = wi;
    if ((uint32_t)DeeployOptNetwork_outputs[opt_w_out] >= 0x10000000u &&
        (uint32_t)DeeployNetwork_inputs[train_w_idx] >= 0x10000000u) {
      memcpy(DeeployNetwork_inputs[train_w_idx],
             DeeployOptNetwork_outputs[opt_w_out],
             DeeployNetwork_inputs_bytes[train_w_idx]);
    }
#endif
  }
#endif /* TRAINING_NUM_WEIGHT_INPUTS */
}

/* Numerical comparison helpers — run on cluster (FC has no FPU) */

typedef struct {
  float    *computed;
  float    *reference;
  uint32_t  n;
  uint32_t *err_count;
} LossCompareArgs;

static void CompareLossesOnCluster(void *args) {
  if (pi_core_id() != 0) return;
  LossCompareArgs *a = (LossCompareArgs *)args;
  float tol = TRAINING_TOLERANCE_ABS;  /* read on cluster — has FPU */
  uint32_t errors = 0;
  for (uint32_t i = 0; i < a->n; i++) {
    float diff = a->computed[i] - a->reference[i];
    if (diff < 0.0f) diff = -diff;
    printf("  [loss %u] computed=%.6f  ref=%.6f  diff=%.6f  TOL=%.6f\r\n",
             i, (double)a->computed[i], (double)a->reference[i],
             (double)diff, (double)tol);
    if (diff > tol) {
      errors++;
    }
  }
  *a->err_count = errors;
}

/* State comparison: compare a computed buffer against a reference array. */
typedef struct {
  float    *computed;
  float    *reference;
  uint32_t  n_elems;
  uint32_t  tensor_idx;
  const char *label;       /* "W", "V", or "H" */
  uint32_t *err_count;
} StateCompareArgs;

static void CompareStateOnCluster(void *args) {
  if (pi_core_id() != 0) return;
  StateCompareArgs *a = (StateCompareArgs *)args;
  float tol = TRAINING_TOLERANCE_ABS;
  uint32_t errors = 0;
  float max_diff = 0.0f;
  for (uint32_t i = 0; i < a->n_elems; i++) {
    float diff = a->computed[i] - a->reference[i];
    if (diff < 0.0f) diff = -diff;
    if (diff > max_diff) max_diff = diff;
    if (diff > tol) {
      printf("    [%s_%u][%u] actual=%.8f ref=%.8f diff=%.8f\r\n",
             a->label, a->tensor_idx, i,
             (double)a->computed[i], (double)a->reference[i], (double)diff);
      errors++;
    }
  }
  printf("  [%s_%u] %u elems, max_diff=%.8f, errors=%u\r\n",
         a->label, a->tensor_idx, a->n_elems, (double)max_diff, errors);
  *a->err_count += errors;
}

/* main */

int main(void) {

printf("=== Siracusa Training Harness (Phase 2 — with OptimizerNetwork) ===\r\n");
printf("N_TRAIN_STEPS=%u  N_ACCUM_STEPS=%u  DATA_INPUTS=%u\r\n",
        (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS,
        (unsigned)TRAINING_NUM_DATA_INPUTS);

//   /* ------------------------------------------------------------------
//    * Cluster bring-up
//    * ------------------------------------------------------------------ */

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

  struct pi_cluster_task cluster_task;

  /* ------------------------------------------------------------------
   * Init training network
   * ------------------------------------------------------------------ */

  printf("Initializing TrainingNetwork...\r\n");
  pi_cluster_task(&cluster_task, InitTrainingNetwork, NULL);
  cluster_task.stack_size       = MAINSTACKSIZE;
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  /* ------------------------------------------------------------------
   * Zero-initialise gradient accumulation buffers.
   * ------------------------------------------------------------------ */

for (uint32_t _gi = 0; _gi < (uint32_t)TRAINING_NUM_GRAD_INPUTS; _gi++) {
  uint32_t _idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + _gi;
  if ((uint32_t)DeeployNetwork_inputs[_idx] >= 0x10000000u) {
    memset(DeeployNetwork_inputs[_idx], 0, DeeployNetwork_inputs_bytes[_idx]);
  }
}

  /* ------------------------------------------------------------------
   * Init optimizer network
   * ------------------------------------------------------------------ */

  printf("Initializing OptimizerNetwork...\r\n");
  pi_cluster_task(&cluster_task, InitOptimizerNetwork, NULL);
  cluster_task.stack_size       = MAINSTACKSIZE;
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  /* Initialise Adam state (V, H, R, T) if using Adam optimizer. */
  init_optimizer_state();

//   /* ------------------------------------------------------------------
//    * lazy_reset_grad is the last input of the training network.
//    * ------------------------------------------------------------------ */

  uint32_t reset_idx = DeeployNetwork_num_inputs - 1;

  /* ------------------------------------------------------------------
   * Copy initial weights into network input buffers.
   * (InitTrainingNetwork only malloc's them; testInitWeights[] holds
   *  the actual starting values from inputs.npz.)
   * ------------------------------------------------------------------ */

#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    if ((uint32_t)DeeployNetwork_inputs[idx] >= 0x10000000u) {
      memcpy(DeeployNetwork_inputs[idx], testInitWeights[wi], DeeployNetwork_inputs_bytes[idx]);
    }
  }
#endif

  printf("Starting training (%u optimizer steps x %u accum steps)...\r\n",
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS);

  /* ------------------------------------------------------------------
   * Benchmark accumulators
   * ------------------------------------------------------------------ */
  uint32_t total_train_cycles  = 0;
  uint32_t total_opt_cycles    = 0;
  uint32_t min_train_cycles    = 0xFFFFFFFFu;
  uint32_t max_train_cycles    = 0;
  uint32_t min_opt_cycles      = 0xFFFFFFFFu;
  uint32_t max_opt_cycles      = 0;

  for (uint32_t update_step = 0; update_step < N_TRAIN_STEPS; update_step++) {

    uint32_t step_train_cycles = 0;

    for (uint32_t accum_step = 0; accum_step < N_ACCUM_STEPS; accum_step++) {

      uint32_t mb = update_step * N_ACCUM_STEPS + accum_step;

      printf("  update %u/%u  accum %u/%u  (mini-batch %u)\r\n",
             update_step + 1, (unsigned)N_TRAIN_STEPS,
             accum_step + 1,  (unsigned)N_ACCUM_STEPS,
             mb);

      /* ① Set lazy_reset_grad. */
      if ((uint32_t)DeeployNetwork_inputs[reset_idx] >= 0x10000000) {
        *((uint8_t *)DeeployNetwork_inputs[reset_idx]) =
            (accum_step == 0) ? 1u : 0u;
      }

      /* ② Load this mini-batch's data + labels (cycle through unique samples). */
      for (uint32_t buf = 0; buf < TRAINING_NUM_DATA_INPUTS; buf++) {
        if ((uint32_t)DeeployNetwork_inputs[buf] >= 0x10000000) {
          memcpy(DeeployNetwork_inputs[buf],
                 testDataVector[mb % TRAINING_DATA_SIZE][buf],
                 DeeployNetwork_inputs_bytes[buf]);
        }
      }

      /* ③ Forward + backward + InPlaceAccumulatorV2 (measured). */
      ResetTimer();
      StartTimer();

      pi_cluster_task(&cluster_task, RunTrainingNetwork, NULL);
      cluster_task.stack_size       = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      StopTimer();
      uint32_t train_cyc = getCycles();
      step_train_cycles += train_cyc;

      printf("  [BENCH] mb=%u train_cycles=%u\r\n", mb, (unsigned)train_cyc);

      /* ④ Store loss — use memcpy to avoid float registers on FC (no FPU). */
      if ((uint32_t)DeeployNetwork_outputs[0] >= 0x10000000u) {
        memcpy(&stored_losses[mb], DeeployNetwork_outputs[0], sizeof(float));
      }

    } /* end accum_step loop */

    /* ⑤ Optimizer weight update (SGD or Adam) via Deeploy-compiled OptimizerNetwork. */
    uint32_t opt_cyc = 0;
    run_optimizer_step(&opt_cyc);

    printf("  [BENCH] step=%u step_train_cycles=%u opt_cycles=%u\r\n",
           update_step, (unsigned)step_train_cycles, (unsigned)opt_cyc);

    total_train_cycles += step_train_cycles;
    total_opt_cycles   += opt_cyc;
    if (step_train_cycles < min_train_cycles) min_train_cycles = step_train_cycles;
    if (step_train_cycles > max_train_cycles) max_train_cycles = step_train_cycles;
    if (opt_cyc < min_opt_cycles) min_opt_cycles = opt_cyc;
    if (opt_cyc > max_opt_cycles) max_opt_cycles = opt_cyc;

  } /* end update_step loop */

  /* ------------------------------------------------------------------
   * Benchmark summary
   * ------------------------------------------------------------------ */
  uint32_t total_cycles = total_train_cycles + total_opt_cycles;
  uint32_t avg_train = (N_TRAIN_STEPS > 0) ? total_train_cycles / N_TRAIN_STEPS : 0;
  uint32_t avg_opt   = (N_TRAIN_STEPS > 0) ? total_opt_cycles   / N_TRAIN_STEPS : 0;

  /* ------------------------------------------------------------------
   * Compute data transfer bytes per step (FC-side memcpy overhead)
   * ------------------------------------------------------------------ */
  uint32_t data_load_bytes_per_mb = 0;
  for (uint32_t buf = 0; buf < TRAINING_NUM_DATA_INPUTS; buf++) {
    data_load_bytes_per_mb += DeeployNetwork_inputs_bytes[buf];
  }
  uint32_t data_load_bytes_per_step = data_load_bytes_per_mb * N_ACCUM_STEPS;

  uint32_t opt_copy_bytes_per_step = 0;
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t train_g_idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + wi;
    /* Step A: weight + grad → optimizer inputs */
    opt_copy_bytes_per_step += DeeployNetwork_inputs_bytes[train_w_idx];
    opt_copy_bytes_per_step += DeeployNetwork_inputs_bytes[train_g_idx];
    /* Step C: optimizer outputs → training weights */
    opt_copy_bytes_per_step += DeeployNetwork_inputs_bytes[train_w_idx];
#ifdef OPTIMIZER_ADAM
    /* Adam: also copy V_new and H_new back */
    opt_copy_bytes_per_step += DeeployNetwork_inputs_bytes[train_w_idx] * 2;
#endif
  }
#endif

  printf("\r\n[BENCH] ============ BENCHMARK SUMMARY ============\r\n");
  printf("[BENCH] N_TRAIN_STEPS=%u  N_ACCUM_STEPS=%u\r\n",
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS);
#ifdef OPTIMIZER_ADAM
  printf("[BENCH] Optimizer: Adam\r\n");
#else
  printf("[BENCH] Optimizer: SGD\r\n");
#endif
  printf("[BENCH] --- Training (fwd+bwd) per step (=%u mini-batches) ---\r\n",
         (unsigned)N_ACCUM_STEPS);
  printf("[BENCH]   total  = %u cycles\r\n", (unsigned)total_train_cycles);
  printf("[BENCH]   avg    = %u cycles/step\r\n", (unsigned)avg_train);
  printf("[BENCH]   min    = %u cycles/step\r\n", (unsigned)min_train_cycles);
  printf("[BENCH]   max    = %u cycles/step\r\n", (unsigned)max_train_cycles);
  printf("[BENCH] --- Optimizer kernel per step ---\r\n");
  printf("[BENCH]   total  = %u cycles\r\n", (unsigned)total_opt_cycles);
  printf("[BENCH]   avg    = %u cycles/step\r\n", (unsigned)avg_opt);
  printf("[BENCH]   min    = %u cycles/step\r\n", (unsigned)min_opt_cycles);
  printf("[BENCH]   max    = %u cycles/step\r\n", (unsigned)max_opt_cycles);
  printf("[BENCH] --- Data transfer (FC-side memcpy, per step) ---\r\n");
  printf("[BENCH]   data_load     = %u bytes/step (%u bytes/mb x %u mb)\r\n",
         (unsigned)data_load_bytes_per_step,
         (unsigned)data_load_bytes_per_mb, (unsigned)N_ACCUM_STEPS);
  printf("[BENCH]   opt_copy      = %u bytes/step (w+g→opt + opt→w)\r\n",
         (unsigned)opt_copy_bytes_per_step);
  printf("[BENCH]   total_memcpy  = %u bytes/step\r\n",
         (unsigned)(data_load_bytes_per_step + opt_copy_bytes_per_step));
  printf("[BENCH] --- Total ---\r\n");
  printf("[BENCH]   train+opt = %u cycles\r\n", (unsigned)total_cycles);
  printf("[BENCH]   train%%    = %u%%\r\n",
         total_cycles > 0 ? (unsigned)(total_train_cycles / (total_cycles / 100u)) : 0);
  printf("[BENCH]   opt%%      = %u%%\r\n",
         total_cycles > 0 ? (unsigned)(total_opt_cycles / (total_cycles / 100u)) : 0);
  printf("[BENCH] =======================================\r\n\r\n");

  /* ------------------------------------------------------------------
   * Numerical verification — run on cluster (FC has no FPU)
   * ------------------------------------------------------------------ */

  uint32_t loss_err_count = 0;
  uint32_t total_loss_checks = (TOTAL_FWD_PASSES < N_LOSS_REFS) ? TOTAL_FWD_PASSES : N_LOSS_REFS;
  LossCompareArgs loss_cmp_args = {
    .computed  = stored_losses,
    .reference = (float *)testLossRef,
    .n         = total_loss_checks,
    .err_count = &loss_err_count,
  };
  pi_cluster_task(&cluster_task, CompareLossesOnCluster, &loss_cmp_args);
  cluster_task.stack_size       = MAINSTACKSIZE;
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  printf("Loss errors: %u out of %u\r\n", (unsigned)loss_err_count, (unsigned)total_loss_checks);

  /* ------------------------------------------------------------------
   * State verification: final weights (and V/H for Adam)
   * ------------------------------------------------------------------ */
  uint32_t state_err_count = 0;

#if defined(NUM_STATE_REFS) && (NUM_STATE_REFS > 0) && defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  printf("\nComparing final weight states (%u tensors):\r\n", (unsigned)NUM_STATE_REFS);
  for (uint32_t wi = 0; wi < (uint32_t)NUM_STATE_REFS && wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    StateCompareArgs w_args = {
      .computed   = (float *)DeeployNetwork_inputs[train_w_idx],
      .reference  = testWeightRef[wi],
      .n_elems    = testStateRefSizes[wi],
      .tensor_idx = wi,
      .label      = "W",
      .err_count  = &state_err_count,
    };
    pi_cluster_task(&cluster_task, CompareStateOnCluster, &w_args);
    cluster_task.stack_size       = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  }

#if defined(OPTIMIZER_ADAM) && defined(HAS_ADAM_STATE_REFS) && (HAS_ADAM_STATE_REFS > 0)
  printf("\nComparing Adam V states (%u tensors):\r\n", (unsigned)NUM_STATE_REFS);
  for (uint32_t wi = 0; wi < (uint32_t)NUM_STATE_REFS && wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t opt_v_in = OPT_V_IDX(wi);
    StateCompareArgs v_args = {
      .computed   = (float *)DeeployOptNetwork_inputs[opt_v_in],
      .reference  = testVRef[wi],
      .n_elems    = testStateRefSizes[wi],
      .tensor_idx = wi,
      .label      = "V",
      .err_count  = &state_err_count,
    };
    pi_cluster_task(&cluster_task, CompareStateOnCluster, &v_args);
    cluster_task.stack_size       = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  }

  printf("\nComparing Adam H states (%u tensors):\r\n", (unsigned)NUM_STATE_REFS);
  for (uint32_t wi = 0; wi < (uint32_t)NUM_STATE_REFS && wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t opt_h_in = OPT_H_IDX(wi);
    StateCompareArgs h_args = {
      .computed   = (float *)DeeployOptNetwork_inputs[opt_h_in],
      .reference  = testHRef[wi],
      .n_elems    = testStateRefSizes[wi],
      .tensor_idx = wi,
      .label      = "H",
      .err_count  = &state_err_count,
    };
    pi_cluster_task(&cluster_task, CompareStateOnCluster, &h_args);
    cluster_task.stack_size       = MAINSTACKSIZE;
    cluster_task.slave_stack_size = SLAVESTACKSIZE;
    pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  }
#endif /* OPTIMIZER_ADAM && HAS_ADAM_STATE_REFS */

  printf("State errors: %u\r\n", (unsigned)state_err_count);
#endif /* NUM_STATE_REFS */

  uint32_t total_errors = loss_err_count + state_err_count;
  printf("\nErrors: %u out of %u\r\n", (unsigned)total_errors, (unsigned)total_loss_checks);

  return 0;

}
