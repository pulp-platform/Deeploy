/*
 * SPDX-FileCopyrightText: 2020 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

/*
 * Training harness for Siracusa — Phase 2 (with Deeploy-compiled OptimizerNetwork)
 *
 * Loop structure:
 *
 *   InitTrainingNetwork()
 *   InitOptimizerNetwork()
 *   Connect optimizer buffers → training network's weight/grad buffers
 *
 *   for update_step in [0, N_TRAIN_STEPS):          // optimizer steps
 *       for accum_step in [0, N_ACCUM_STEPS):        // mini-batches per update
 *           lazy_reset_grad = (accum_step == 0)      // reset on first, accumulate on rest
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
 *
 * Optimizer buffer layout in DeeployOptNetwork_inputs[] (interleaved pairs):
 *   [2*i]   weight_i     (copied from DeeployNetwork_inputs[TRAINING_NUM_DATA_INPUTS+i])
 *   [2*i+1] grad_acc_i   (copied from DeeployNetwork_inputs[TRAINING_GRAD_BUF_START_IDX+i])
 * DeeployOptNetwork_outputs[i] = weight_i_updated
 *   → copied back to DeeployNetwork_inputs[TRAINING_NUM_DATA_INPUTS+i]
 *
 * Compile-time constants (emitted by code generator into testinputs.h):
 *   N_TRAIN_STEPS              number of optimizer (weight-update) steps
 *   N_ACCUM_STEPS              number of mini-batches accumulated per update
 *   TRAINING_NUM_DATA_INPUTS   inputs that change each mini-batch (data + labels)
 *   TRAINING_GRAD_BUF_START_IDX  first grad acc buffer index in DeeployNetwork_inputs[]
 *   TRAINING_NUM_GRAD_INPUTS   number of grad accumulation buffers (== number of weights)
 *   TRAINING_NUM_WEIGHT_INPUTS number of trainable weight buffers
 *   TRAINING_LEARNING_RATE     SGD learning rate (for reference — embedded in optimizer ONNX)
 *
 * Reference comparison constants (emitted into testoutputs.h):
 *   N_LOSS_REFS                number of reference loss values
 *   NUM_WEIGHT_REFS            number of reference weight tensors
 *   TRAINING_TOLERANCE_ABS     absolute comparison tolerance
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

#define MAINSTACKSIZE  8000
#define SLAVESTACKSIZE 3800

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
 * Optimizer buffer connection
 *
 * Connect DeeployOptNetwork_inputs[]/outputs[] to the training network's
 * weight and grad acc buffers via memcpy.
 *
 * Optimizer ONNX input order: [w0, g0, w1, g1, ...]  (interleaved pairs)
 * Optimizer ONNX output order: [w0_updated, w1_updated, ...]
 * ---------------------------------------------------------------------- */

static void connect_optimizer_buffers(void) {
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  /* Nothing to pre-allocate — InitOptimizerNetwork() already allocated the
   * optimizer's static buffers and set DeeployOptNetwork_inputs[]/outputs[].
   * We only need to sync data at each optimizer step (see run_optimizer_step). */
  (void)0;
#endif
}

static void run_optimizer_step(void) {
#if defined(TRAINING_NUM_WEIGHT_INPUTS) && (TRAINING_NUM_WEIGHT_INPUTS > 0)
  /* --- Step A: copy current weights + grad acc → optimizer input buffers --- */
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t train_g_idx = (uint32_t)TRAINING_GRAD_BUF_START_IDX + wi;
    uint32_t opt_w_in    = 2u * wi;
    uint32_t opt_g_in    = 2u * wi + 1u;

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


  struct pi_cluster_task opt_task;
  pi_cluster_task(&opt_task, RunOptimizerNetwork, NULL);
  opt_task.stack_size       = MAINSTACKSIZE;
  opt_task.slave_stack_size = SLAVESTACKSIZE;
  pi_cluster_send_task_to_cl(&cluster_dev, &opt_task);

  /* --- Step C: copy weight_updated back to training network's weight buffers --- */
  for (uint32_t wi = 0; wi < (uint32_t)TRAINING_NUM_WEIGHT_INPUTS; wi++) {
    uint32_t train_w_idx  = (uint32_t)TRAINING_NUM_DATA_INPUTS + wi;
    uint32_t opt_w_out    = wi;

    if ((uint32_t)DeeployOptNetwork_outputs[opt_w_out] >= 0x10000000u &&
        (uint32_t)DeeployNetwork_inputs[train_w_idx] >= 0x10000000u) {
      memcpy(DeeployNetwork_inputs[train_w_idx],
             DeeployOptNetwork_outputs[opt_w_out],
             DeeployNetwork_inputs_bytes[train_w_idx]);
    }
  }
#endif /* TRAINING_NUM_WEIGHT_INPUTS */
}

/* -------------------------------------------------------------------------
 * Numerical comparison helpers — run on cluster (FC has no FPU)
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
  float tol = TRAINING_TOLERANCE_ABS;  /* read on cluster — has FPU */
  uint32_t errors = 0;
  for (uint32_t i = 0; i < a->n; i++) {
    float diff = a->computed[i] - a->reference[i];
    if (diff < 0.0f) diff = -diff;
    if (diff > tol) {
      errors++;
      printf("  [loss %u] computed=%.6f  ref=%.6f  diff=%.6f  TOL=%.6f\r\n",
             i, (double)a->computed[i], (double)a->reference[i],
             (double)diff, (double)tol);
    }
  }
  *a->err_count = errors;
}

/* -------------------------------------------------------------------------
 * main
 * ---------------------------------------------------------------------- */

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

//   connect_optimizer_buffers();

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

  uint32_t training_cycles   = 0;
  uint32_t optimizer_cycles  = 0;

  for (uint32_t update_step = 0; update_step < N_TRAIN_STEPS; update_step++) {

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

      /* ② Load this mini-batch's data + labels. */
      for (uint32_t buf = 0; buf < TRAINING_NUM_DATA_INPUTS; buf++) {
        if ((uint32_t)DeeployNetwork_inputs[buf] >= 0x10000000) {
          memcpy(DeeployNetwork_inputs[buf],
                 testDataVector[mb][buf],
                 DeeployNetwork_inputs_bytes[buf]);
        }
      }

      /* ③ Forward + backward + InPlaceAccumulatorV2. */
      pi_cluster_task(&cluster_task, RunTrainingNetwork, NULL);
      cluster_task.stack_size       = MAINSTACKSIZE;
      cluster_task.slave_stack_size = SLAVESTACKSIZE;
      pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);

      /* ④ Store loss — use memcpy to avoid float registers on FC (no FPU). */
      if ((uint32_t)DeeployNetwork_outputs[0] >= 0x10000000u) {
        memcpy(&stored_losses[mb], DeeployNetwork_outputs[0], sizeof(float));
      }

    } /* end accum_step loop */

    /* ⑤ SGD weight update via Deeploy-compiled OptimizerNetwork. */
    run_optimizer_step();

  } /* end update_step loop */

  // printf("Training complete.\r\n");
  // printf("Total training cycles  : %u\r\n", training_cycles);
  // printf("Total optimizer cycles : %u\r\n", optimizer_cycles);


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
  printf("Errors: %u out of %u\r\n", (unsigned)loss_err_count, (unsigned)total_loss_checks);



  return 0;

}
