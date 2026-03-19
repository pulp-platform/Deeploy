/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Training harness for GAP9 — Phase 2 (with Deeploy-compiled OptimizerNetwork)
 *
 * Adapted from Siracusa training harness for GAP9 platform.
 *
 * Loop structure:
 *
 *   InitTrainingNetwork()
 *   InitOptimizerNetwork()
 *
 *   for update_step in [0, N_TRAIN_STEPS):          // optimizer steps
 *       for accum_step in [0, N_ACCUM_STEPS):        // mini-batches per update
 *           lazy_reset_grad = (accum_step == 0)
 *           load data for this mini-batch
 *           RunTrainingNetwork()                     // fwd + bwd + InPlaceAccumulatorV2
 *           store loss value
 *       // SGD weight update via Deeploy-compiled optimizer kernel:
 *       copy weights + grad_acc → optimizer input buffers
 *       RunOptimizerNetwork()
 *       copy weight_updated ← optimizer output buffers → training weight buffers
 *
 *   Numerical verification:
 *     - Compare stored loss values against testLossRef[] (from testoutputs.h)
 *
 * Buffer layout in DeeployNetwork_inputs[] (must match ONNX input order):
 *   [0 .. TRAINING_NUM_DATA_INPUTS-1]              data + labels (per mini-batch)
 *   [TRAINING_NUM_DATA_INPUTS ..
 *    .. TRAINING_GRAD_BUF_START_IDX-1]             weights (persistent)
 *   [TRAINING_GRAD_BUF_START_IDX ..
 *    .. +TRAINING_NUM_GRAD_INPUTS-1]               grad accumulation bufs (persistent)
 *   [DeeployNetwork_num_inputs-1]                  lazy_reset_grad uint8
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

/* Helper: true when ptr is in L2 (CPU-accessible); false when in L3 (external RAM) */
#define IS_L2(ptr)  ((uint32_t)(ptr) >= 0x10000000u)

/* -------------------------------------------------------------------------
 * Compile-time defaults — override via CMake target_compile_definitions
 * ---------------------------------------------------------------------- */

#ifndef N_TRAIN_STEPS
#define N_TRAIN_STEPS 1
#endif

#ifndef N_ACCUM_STEPS
#define N_ACCUM_STEPS 1
#endif

#ifndef TRAINING_NUM_DATA_INPUTS
#define TRAINING_NUM_DATA_INPUTS 2
#endif

/* Training networks are much larger than inference; the master core needs
 * a bigger stack for the generated RunTrainingNetwork/InitTrainingNetwork
 * functions which have many local variables across deep closure chains. */
#define MAINSTACKSIZE  12000
#define SLAVESTACKSIZE 6000

/* -------------------------------------------------------------------------
 * Cluster device
 * ---------------------------------------------------------------------- */

struct pi_device cluster_dev;

/* -------------------------------------------------------------------------
 * Loss storage (one value per forward pass)
 * ---------------------------------------------------------------------- */

#define TOTAL_FWD_PASSES (N_TRAIN_STEPS * N_ACCUM_STEPS)
static float stored_losses[TOTAL_FWD_PASSES];

/* -------------------------------------------------------------------------
 * Wrapper functions for cluster task dispatch.
 *
 * GAP9 code generator produces functions with
 *   (uint32_t core_id, uint32_t numThreads)
 * parameters, while pi_cluster_task expects void (*)(void*).
 * ---------------------------------------------------------------------- */

void InitTrainingNetworkWrapper(void *args) {
  (void)args;
  InitTrainingNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

void RunTrainingNetworkWrapper(void *args) {
  (void)args;
  RunTrainingNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

void InitOptimizerNetworkWrapper(void *args) {
  (void)args;
  InitOptimizerNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

void RunOptimizerNetworkWrapper(void *args) {
  (void)args;
  RunOptimizerNetwork(pi_core_id(), pi_cl_cluster_nb_cores());
}

/* -------------------------------------------------------------------------
 * L3-aware memory transfer: handles all combinations of L2/L3 src and dst
 * ---------------------------------------------------------------------- */

static void l3_aware_copy(void *dst, const void *src, uint32_t bytes) {
  if (IS_L2(dst) && IS_L2(src)) {
    memcpy(dst, src, bytes);
  } else if (IS_L2(dst)) {
    /* L3 → L2 */
    ram_read(dst, (void *)src, bytes);
  } else if (IS_L2(src)) {
    /* L2 → L3 */
    ram_write(dst, (void *)src, bytes);
  } else {
    /* L3 → L3: stage through a temporary L2 buffer */
    void *tmp = pi_l2_malloc(bytes);
    ram_read(tmp, (void *)src, bytes);
    ram_write(dst, tmp, bytes);
    pi_l2_free(tmp, bytes);
  }
}

/* -------------------------------------------------------------------------
 * Optimizer step: copy buffers → run → copy back
 * ---------------------------------------------------------------------- */

static void run_optimizer_step(void) {
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  /* --- Step A: copy current weights + grad acc → optimizer input buffers --- */
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t train_g_idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + wi;
    uint32_t opt_w_in    = 2u * wi;
    uint32_t opt_g_in    = 2u * wi + 1u;

    l3_aware_copy(DeeployOptNetwork_inputs[opt_w_in],
                  DeeployNetwork_inputs[train_w_idx],
                  DeeployOptNetwork_inputs_bytes[opt_w_in]);
    l3_aware_copy(DeeployOptNetwork_inputs[opt_g_in],
                  DeeployNetwork_inputs[train_g_idx],
                  DeeployOptNetwork_inputs_bytes[opt_g_in]);
  }

  /* --- Step B: Run optimizer network --- */
  struct pi_cluster_task opt_task;
  pi_cluster_task(&opt_task, RunOptimizerNetworkWrapper, NULL);
  opt_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &opt_task);

  /* --- Step C: copy weight_updated back to training network --- */
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx  = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t opt_w_out    = wi;

    uint32_t opt_bytes   = DeeployOptNetwork_outputs_bytes[opt_w_out];
    uint32_t train_bytes = DeeployNetwork_inputs_bytes[train_w_idx];
    if (opt_bytes == train_bytes) {
      l3_aware_copy(DeeployNetwork_inputs[train_w_idx],
                    DeeployOptNetwork_outputs[opt_w_out],
                    opt_bytes);
    } else {
      /* Broadcasted bias: fill every tile with updated value. */
      for (uint32_t off = 0; off < train_bytes; off += opt_bytes) {
        uint32_t chunk = (off + opt_bytes <= train_bytes) ? opt_bytes : (train_bytes - off);
        l3_aware_copy((char *)DeeployNetwork_inputs[train_w_idx] + off,
                      DeeployOptNetwork_outputs[opt_w_out],
                      chunk);
      }
    }
  }
#endif /* TRAINING_NUM_WEIGHT_INPUTS */
}

/* -------------------------------------------------------------------------
 * Numerical comparison — run on cluster (FC has no FPU)
 * ---------------------------------------------------------------------- */

typedef struct {
  float    *computed;
  float    *reference;
  uint32_t  n;
  uint32_t *err_count;
} LossCompareArgs;

static void CompareLossesOnCluster(void *args) {
  if (pi_core_id() != 0) return;
  LossCompareArgs *a = (LossCompareArgs *)args;
  float tol = TRAINING_TOLERANCE_ABS;
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

static void CL_CompareLosses(void *arg) {
  pi_cl_team_fork(NUM_CORES, CompareLossesOnCluster, arg);
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

int main(void) {

  printf("=== GAP9 Training Harness (Phase 2 — with OptimizerNetwork) ===\r\n");
  printf("N_TRAIN_STEPS=%u  N_ACCUM_STEPS=%u  DATA_INPUTS=%u\r\n",
          (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS,
          (unsigned)TRAINING_NUM_DATA_INPUTS);

  /* ------------------------------------------------------------------
   * Cluster bring-up
   * ------------------------------------------------------------------ */

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
  pi_cluster_task(&cluster_task, InitTrainingNetworkWrapper, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  /* ------------------------------------------------------------------
   * Zero-initialise gradient accumulation buffers.
   * ------------------------------------------------------------------ */

  for (uint32_t _gi = 0; _gi < (uint32_t)TRAINING_NUM_GRAD_INPUTS; _gi++) {
    uint32_t _idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + _gi;
    uint32_t bytes = DeeployNetwork_inputs_bytes[_idx];
    void *buf = DeeployNetwork_inputs[_idx];
    if (IS_L2(buf)) {
      memset(buf, 0, bytes);
    } else {
      /* Write zeros into L3 via DMA using a temporary L2 zero page */
      uint8_t *zero_page = pi_l2_malloc(512);
      memset(zero_page, 0, 512);
      for (uint32_t off = 0; off < bytes; off += 512) {
        uint32_t chunk = (off + 512 <= bytes) ? 512 : (bytes - off);
        ram_write((char *)buf + off, zero_page, chunk);
      }
      pi_l2_free(zero_page, 512);
    }
  }

  /* ------------------------------------------------------------------
   * Init optimizer network
   * ------------------------------------------------------------------ */

  printf("Initializing OptimizerNetwork...\r\n");
  pi_cluster_task(&cluster_task, InitOptimizerNetworkWrapper, NULL);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

  /* ------------------------------------------------------------------
   * lazy_reset_grad is the last input of the training network.
   * ------------------------------------------------------------------ */

  uint32_t reset_idx = DeeployNetwork_num_inputs - 1;

  /* ------------------------------------------------------------------
   * Copy initial weights into network input buffers.
   * ------------------------------------------------------------------ */

#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    l3_aware_copy(DeeployNetwork_inputs[idx], testInitWeights[wi], DeeployNetwork_inputs_bytes[idx]);
  }
#endif

  printf("Starting training (%u optimizer steps x %u accum steps)...\r\n",
         (unsigned)N_TRAIN_STEPS, (unsigned)N_ACCUM_STEPS);

  for (uint32_t update_step = 0; update_step < N_TRAIN_STEPS; update_step++) {

    for (uint32_t accum_step = 0; accum_step < N_ACCUM_STEPS; accum_step++) {

      uint32_t mb = update_step * N_ACCUM_STEPS + accum_step;

      printf("  update %u/%u  accum %u/%u  (mini-batch %u)\r\n",
             update_step + 1, (unsigned)N_TRAIN_STEPS,
             accum_step + 1,  (unsigned)N_ACCUM_STEPS,
             mb);

      /* 1. Set lazy_reset_grad. */
      {
        void *reset_ptr = DeeployNetwork_inputs[reset_idx];
        uint8_t reset_val = (accum_step == 0) ? 1u : 0u;
        if (IS_L2(reset_ptr)) {
          *((uint8_t *)reset_ptr) = reset_val;
        } else {
          ram_write(reset_ptr, &reset_val, sizeof(uint8_t));
        }
      }

      /* 2. Load this mini-batch's data + labels. */
      for (uint32_t buf = 0; buf < TRAINING_NUM_DATA_INPUTS; buf++) {
        l3_aware_copy(DeeployNetwork_inputs[buf],
                      testDataVector[mb % TRAINING_DATA_SIZE][buf],
                      DeeployNetwork_inputs_bytes[buf]);
      }

      /* 3. Forward + backward + InPlaceAccumulatorV2. */
      pi_cluster_task(&cluster_task, RunTrainingNetworkWrapper, NULL);
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      /* 4. Store loss. */
      {
        void *loss_ptr = DeeployNetwork_outputs[0];
        if (IS_L2(loss_ptr)) {
          memcpy(&stored_losses[mb], loss_ptr, sizeof(float));
        } else {
          ram_read(&stored_losses[mb], loss_ptr, sizeof(float));
        }
      }

    } /* end accum_step loop */

    /* 5. SGD weight update via Deeploy-compiled OptimizerNetwork. */
    run_optimizer_step();

  } /* end update_step loop */

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
  pi_cluster_task(&cluster_task, CL_CompareLosses, &loss_cmp_args);
  cluster_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
  printf("Errors: %u out of %u\r\n", (unsigned)loss_err_count, (unsigned)total_loss_checks);

  return 0;
}
