#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "TrainingNetwork.h"

float32_t *DeeployNetwork_input_0;
uint8_t *DeeployNetwork_input_1;
float32_t *DeeployNetwork_input_2;
float32_t *DeeployNetwork_input_3;
float32_t *DeeployNetwork_input_4;
float32_t *DeeployNetwork_input_5;
uint8_t *DeeployNetwork_input_6;
float32_t *DeeployNetwork_output_0;
float32_t *DeeployNetwork_output_1;
float32_t *DeeployNetwork_output_2;

static PI_L2 float32_t DeeployNetwork_fc1_bias_tensor[8] = {0.01703643798828125f,    0.022536905482411385f, 0.0042967586778104305f, 0.021887388080358505f,
                                                            -0.0030818157829344273f, -0.03479762002825737f, 0.01098635420203209f,   0.011764267459511757f};

static PI_L2 float32_t DeeployNetwork_fc2_bias_tensor[10] = {-0.2502647042274475f,   0.30938681960105896f,   0.11061073839664459f, -0.06800693273544312f,
                                                             0.10814286768436432f,   0.0006021520239301026f, 0.10259895771741867f, 0.25649648904800415f,
                                                             -0.018703928217291832f, 0.26217177510261536f};

void *DeeployNetwork_inputs[7];
void *DeeployNetwork_outputs[3];
extern struct pi_device cluster_dev;
typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
} _node_1_fc1_Gemm_Gemm_tiling_closure_args_t;

static void _node_1_fc1_Gemm_Gemm_tiling_closure(void *_node_1_fc1_Gemm_Gemm_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_Gemm_tiling_closure_args_t *args = (_node_1_fc1_Gemm_Gemm_tiling_closure_args_t *)_node_1_fc1_Gemm_Gemm_tiling_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: node_1_fc1_Gemm_Gemm, Op: Gemm)
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_input_0 = DeeployNetwork_input_0;
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_input_2 = DeeployNetwork_input_2;
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_fc1_bias_tensor = DeeployNetwork_fc1_bias_tensor;
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor;

  for (uint32_t i = 0; i < 1.0; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_input_0,
                                  ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_input_2,
                                  ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_fc1_bias_tensor,
                                  ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor, 1, 784, 8, 0, 1);
    ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_input_0 += 1 * 784;

    ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_input_2 += 784 * 8;

    ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_fc1_bias_tensor += 1 * 8;

    ref_DeeployNetwork_node_0_fc1_Gemm__0_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor += 1 * 8;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
} _node_1_fc1_Gemm_Gemm_cluster_fork_args_t;

static void _node_1_fc1_Gemm_Gemm_cluster_fork(void *_node_1_fc1_Gemm_Gemm_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_Gemm_cluster_fork_args_t *args = (_node_1_fc1_Gemm_Gemm_cluster_fork_args_t *)_node_1_fc1_Gemm_Gemm_cluster_fork_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;

  // CLOSURE FUNCTION CALL
  _node_1_fc1_Gemm_Gemm_tiling_closure_args_t DeeployNetwork__node_1_fc1_Gemm_Gemm_tiling_closure_args =
      (_node_1_fc1_Gemm_Gemm_tiling_closure_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor};

  // _node_1_fc1_Gemm_Gemm_tiling_closure CLOSURE CALL
  _node_1_fc1_Gemm_Gemm_tiling_closure(&DeeployNetwork__node_1_fc1_Gemm_Gemm_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
} _node_1_fc1_Gemm_Gemm_closure_args_t;

static void _node_1_fc1_Gemm_Gemm_closure(void *_node_1_fc1_Gemm_Gemm_closure_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_Gemm_closure_args_t *args = (_node_1_fc1_Gemm_Gemm_closure_args_t *)_node_1_fc1_Gemm_Gemm_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;

  // CLOSURE FUNCTION CALL
  _node_1_fc1_Gemm_Gemm_cluster_fork_args_t DeeployNetwork__node_1_fc1_Gemm_Gemm_cluster_fork_args =
      (_node_1_fc1_Gemm_Gemm_cluster_fork_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_node_1_fc1_Gemm_Gemm_cluster_fork, &DeeployNetwork__node_1_fc1_Gemm_Gemm_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
} _node_1_fc1_Gemm_Gemm_closure_L3_args_t;

static void _node_1_fc1_Gemm_Gemm_closure_L3(void *_node_1_fc1_Gemm_Gemm_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_Gemm_closure_L3_args_t *args = (_node_1_fc1_Gemm_Gemm_closure_L3_args_t *)_node_1_fc1_Gemm_Gemm_closure_L3_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;

  // CLOSURE FUNCTION CALL
  _node_1_fc1_Gemm_Gemm_closure_args_t DeeployNetwork__node_1_fc1_Gemm_Gemm_closure_args =
      (_node_1_fc1_Gemm_Gemm_closure_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor};

  // _node_1_fc1_Gemm_Gemm_closure CLOSURE CALL
  _node_1_fc1_Gemm_Gemm_closure(&DeeployNetwork__node_1_fc1_Gemm_Gemm_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor;
} _node_2_fc2_Gemm_Gemm_tiling_closure_args_t;

static void _node_2_fc2_Gemm_Gemm_tiling_closure(void *_node_2_fc2_Gemm_Gemm_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_Gemm_tiling_closure_args_t *args = (_node_2_fc2_Gemm_Gemm_tiling_closure_args_t *)_node_2_fc2_Gemm_Gemm_tiling_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: node_2_fc2_Gemm_Gemm, Op: Gemm)
  float32_t *ref_DeeployNetwork_output_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *ref_DeeployNetwork_output_tensor_DeeployNetwork_input_3 = DeeployNetwork_input_3;
  float32_t *ref_DeeployNetwork_output_tensor_DeeployNetwork_fc2_bias_tensor = DeeployNetwork_fc2_bias_tensor;
  float32_t *ref_DeeployNetwork_output_tensor_DeeployNetwork_output_tensor = DeeployNetwork_output_tensor;

  for (uint32_t i = 0; i < 1.0; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(ref_DeeployNetwork_output_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor,
                                  ref_DeeployNetwork_output_tensor_DeeployNetwork_input_3, ref_DeeployNetwork_output_tensor_DeeployNetwork_fc2_bias_tensor,
                                  ref_DeeployNetwork_output_tensor_DeeployNetwork_output_tensor, 1, 8, 10, 0, 1);
    ref_DeeployNetwork_output_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor += 1 * 8;

    ref_DeeployNetwork_output_tensor_DeeployNetwork_input_3 += 8 * 10;

    ref_DeeployNetwork_output_tensor_DeeployNetwork_fc2_bias_tensor += 1 * 10;

    ref_DeeployNetwork_output_tensor_DeeployNetwork_output_tensor += 1 * 10;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor;
} _node_2_fc2_Gemm_Gemm_cluster_fork_args_t;

static void _node_2_fc2_Gemm_Gemm_cluster_fork(void *_node_2_fc2_Gemm_Gemm_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_Gemm_cluster_fork_args_t *args = (_node_2_fc2_Gemm_Gemm_cluster_fork_args_t *)_node_2_fc2_Gemm_Gemm_cluster_fork_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_Gemm_tiling_closure_args_t DeeployNetwork__node_2_fc2_Gemm_Gemm_tiling_closure_args = (_node_2_fc2_Gemm_Gemm_tiling_closure_args_t){
      .DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor, .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor};

  // _node_2_fc2_Gemm_Gemm_tiling_closure CLOSURE CALL
  _node_2_fc2_Gemm_Gemm_tiling_closure(&DeeployNetwork__node_2_fc2_Gemm_Gemm_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor;
} _node_2_fc2_Gemm_Gemm_closure_args_t;

static void _node_2_fc2_Gemm_Gemm_closure(void *_node_2_fc2_Gemm_Gemm_closure_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_Gemm_closure_args_t *args = (_node_2_fc2_Gemm_Gemm_closure_args_t *)_node_2_fc2_Gemm_Gemm_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_Gemm_cluster_fork_args_t DeeployNetwork__node_2_fc2_Gemm_Gemm_cluster_fork_args = (_node_2_fc2_Gemm_Gemm_cluster_fork_args_t){
      .DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor, .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_node_2_fc2_Gemm_Gemm_cluster_fork, &DeeployNetwork__node_2_fc2_Gemm_Gemm_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor;
} _node_2_fc2_Gemm_Gemm_closure_L3_args_t;

static void _node_2_fc2_Gemm_Gemm_closure_L3(void *_node_2_fc2_Gemm_Gemm_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_Gemm_closure_L3_args_t *args = (_node_2_fc2_Gemm_Gemm_closure_L3_args_t *)_node_2_fc2_Gemm_Gemm_closure_L3_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_Gemm_closure_args_t DeeployNetwork__node_2_fc2_Gemm_Gemm_closure_args = (_node_2_fc2_Gemm_Gemm_closure_args_t){
      .DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor, .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor};

  // _node_2_fc2_Gemm_Gemm_closure CLOSURE CALL
  _node_2_fc2_Gemm_Gemm_closure(&DeeployNetwork__node_2_fc2_Gemm_Gemm_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
} _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args_t;

static void
_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure(void *_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args_t *)_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;

  // CLOSURE FUNCTION CALL

  BEGIN_SINGLE_CORE
  // SoftmaxCrossEntropyLoss dual-output (Name: onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss, Op: SoftmaxCrossEntropyLoss)
  float32_t sce_total_loss = 0.0f;
  for (uint32_t i = 0; i < 1; i++) {
    float32_t sce_max_logit = DeeployNetwork_output_tensor[i * 10];
    for (uint32_t j = 1; j < 10; j++) {
      if (DeeployNetwork_output_tensor[i * 10 + j] > sce_max_logit)
        sce_max_logit = DeeployNetwork_output_tensor[i * 10 + j];
    }
    float32_t sce_sum_exp = 0.0f;
    for (uint32_t j = 0; j < 10; j++)
      sce_sum_exp += expf(DeeployNetwork_output_tensor[i * 10 + j] - sce_max_logit);
    float32_t sce_log_sum_exp = logf(sce_sum_exp);
    for (uint32_t j = 0; j < 10; j++)
      DeeployNetwork_onnxlog_prob3_tensor[i * 10 + j] = DeeployNetwork_output_tensor[i * 10 + j] - sce_max_logit - sce_log_sum_exp;
    sce_total_loss += -(DeeployNetwork_output_tensor[i * 10 + (uint32_t)(DeeployNetwork_input_1[i])] - sce_max_logit - sce_log_sum_exp);
  }
  DeeployNetwork_output_0[0] = sce_total_loss / (float32_t)1;
  printf("    [SCE] loss=%.6f\r\n", (double)DeeployNetwork_output_0[0]);
  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
} _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args_t;

static void _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork(void *_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args_t *)_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;

  // CLOSURE FUNCTION CALL
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args_t
      DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args =
          (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args_t){
              .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor, .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor};

  // _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure CLOSURE CALL
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure(
      &DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
} _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args_t;

static void _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure(void *_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args_t *)_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;

  // CLOSURE FUNCTION CALL
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args_t
      DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args =
          (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args_t){
              .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor, .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork,
                  &DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
} _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args_t;

static void _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3(void *_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args_t *)_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args;

  float32_t *DeeployNetwork_output_tensor = args->DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;

  // CLOSURE FUNCTION CALL
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args_t DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args =
      (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args_t){.DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
                                                                             .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor};

  // _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure CLOSURE CALL
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure(&DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
} _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args_t;

static void _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure(
    void *_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args_t *)
          _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args;

  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;

  // CLOSURE FUNCTION CALL

  BEGIN_SINGLE_CORE
  // SoftmaxCrossEntropyLossGrad (Name: onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward, Op:
  // SoftmaxCrossEntropyLossGrad)
  float32_t batch_norm = 1.0f / 1;
  for (uint32_t i = 0; i < 1; i++) {
    for (uint32_t j = 0; j < 10; j++) {
      float32_t prob = expf(DeeployNetwork_onnxlog_prob3_tensor[i * 10 + j]);
      if (j == (DeeployNetwork_input_1[i])) {
        DeeployNetwork_output_grad_tensor[i * 10 + j] = (prob - 1.0f) * batch_norm * batch_norm; // RW: one batch_norm for loss norm, one for gradient norm
      } else {
        DeeployNetwork_output_grad_tensor[i * 10 + j] = prob * batch_norm * batch_norm;
      }
    }
  }

  END_SINGLE_CORE

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
} _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args_t;

static void _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork(
    void *_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args_t *)
          _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args;

  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;

  // CLOSURE FUNCTION CALL
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args_t
      DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args =
          (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args_t){
              .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor,
              .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor};

  // _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure CLOSURE CALL
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure(
      &DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
} _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args_t;

static void _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure(
    void *_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args_t *)
          _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args;

  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;

  // CLOSURE FUNCTION CALL
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args_t
      DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args =
          (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args_t){
              .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor,
              .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork,
                  &DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
} _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args_t;

static void _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3(
    void *_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args) {
  // CLOSURE ARG CAST
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args_t *args =
      (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args_t *)
          _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args;

  float32_t *DeeployNetwork_onnxlog_prob3_tensor = args->DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;

  // CLOSURE FUNCTION CALL
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args_t
      DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args =
          (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args_t){
              .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor,
              .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor};

  // _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure CLOSURE CALL
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure(
      &DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args_t;

static void _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure(void *_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args_t *)_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: node_2_fc2_Gemm_GradGemm_1_Gemm_backward, Op: Gemm)
  float32_t *ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor;
  float32_t *ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *ref_DeeployNetwork_fc2_weight_grad_tensor_C = NULL;
  float32_t *ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor;

  for (uint32_t i = 0; i < 1.0; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_output_grad_tensor,
                                  ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor, NULL,
                                  ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_fc2_weight_grad_tensor, 10, 1, 8, 1, 0);
    ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_output_grad_tensor += 10 * 1;

    ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_tensor += 1 * 8;

    ref_DeeployNetwork_fc2_weight_grad_tensor_DeeployNetwork_fc2_weight_grad_tensor += 10 * 8;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args_t;

static void _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork(void *_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args_t *)_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor,
                                                                        .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                        .DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor};

  // _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure CLOSURE CALL
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure(&DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args_t;

static void _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure(void *_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args_t *)_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor,
                                                                      .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                      .DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork,
                  &DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args_t;

static void _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3(void *_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args_t *)_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor,
                                                                 .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                 .DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor};

  // _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure CLOSURE CALL
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure(&DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t;

static void _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure(void *_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t *)_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args;

  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: node_2_fc2_Gemm_GradGemm_0_Gemm_backward, Op: Gemm)
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor;
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_input_3 = DeeployNetwork_input_3;
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_C = NULL;
  float32_t *ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;

  for (uint32_t i = 0; i < 1.0; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_output_grad_tensor,
                                  ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_input_3, NULL,
                                  ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor, 1, 10, 8, 0, 0);
    ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_output_grad_tensor += 1 * 10;

    ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_input_3 += 10 * 8;

    ref_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor += 1 * 8;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t;

static void _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork(void *_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t *)_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args;

  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t){.DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                        .DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor =
                                                                            DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor};

  // _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure CLOSURE CALL
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure(&DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args_t;

static void _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure(void *_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args_t *)_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args;

  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t){.DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                      .DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor =
                                                                          DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork,
                  &DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
} _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t;

static void _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3(void *_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t *args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t *)_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args;

  float32_t *DeeployNetwork_output_grad_tensor = args->DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args_t){.DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                 .DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor =
                                                                     DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor};

  // _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure CLOSURE CALL
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure(&DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t;

static void _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure(void *_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t *args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t *)_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL

  // GEMM (Name: node_1_fc1_Gemm_GradGemm_0_Gemm_backward, Op: Gemm)
  float32_t *ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_input_0 = DeeployNetwork_input_0;
  float32_t *ref_DeeployNetwork_fc1_weight_grad_tensor_C = NULL;
  float32_t *ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor;

  for (uint32_t i = 0; i < 1.0; i++) {
    PULP_Gemm_fp32_fp32_fp32_fp32(ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor,
                                  ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_input_0, NULL,
                                  ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_fc1_weight_grad_tensor, 8, 1, 784, 1, 0);
    ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor += 8 * 1;

    ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_input_0 += 1 * 784;

    ref_DeeployNetwork_fc1_weight_grad_tensor_DeeployNetwork_fc1_weight_grad_tensor += 8 * 784;
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t;

static void _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork(void *_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t *args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t *)_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor =
                                                                            DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor,
                                                                        .DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor};

  // _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure CLOSURE CALL
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure(&DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args_t;

static void _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure(void *_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args_t *args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args_t *)_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor =
                                                                          DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor,
                                                                      .DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork,
                  &DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t;

static void _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3(void *_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args) {
  // CLOSURE ARG CAST
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t *args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t *)_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args;

  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = args->DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args_t DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor,
                                                                 .DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor};

  // _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure CLOSURE CALL
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure(&DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args_t;

static void _GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure(void *_GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args_t *args =
      (_GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args_t *)_GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args;

  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL

  // InPlaceAccumulatorV2 - true in-place (Name: GradientAccumulator2_InPlaceAccumulatorV2_backward, Op: InPlaceAccumulatorV2)
  // Writes result to accum_buffer (in-place) and data_out (explicit output).
  // In training, data_out aliases accum_buffer (same or separate allocation).
  // Reset (lazy_reset_grad=1): accum_buffer  = gradient
  // Accum (lazy_reset_grad=0): accum_buffer += gradient
  int8_t GradientAccumulator2_InPlaceAccumulatorV2_backward_core_id = pi_core_id();
  int8_t GradientAccumulator2_InPlaceAccumulatorV2_backward_log2Core = log2(NUM_CORES);
  int32_t GradientAccumulator2_InPlaceAccumulatorV2_backward_chunk =
      (80 >> GradientAccumulator2_InPlaceAccumulatorV2_backward_log2Core) + ((80 & (NUM_CORES - 1)) != 0);
  int32_t GradientAccumulator2_InPlaceAccumulatorV2_backward_start =
      MIN(GradientAccumulator2_InPlaceAccumulatorV2_backward_chunk * GradientAccumulator2_InPlaceAccumulatorV2_backward_core_id, (int32_t)80);
  int32_t GradientAccumulator2_InPlaceAccumulatorV2_backward_stop =
      MIN(GradientAccumulator2_InPlaceAccumulatorV2_backward_start + GradientAccumulator2_InPlaceAccumulatorV2_backward_chunk, (int32_t)80);

  if (DeeployNetwork_input_6[0]) {
    for (int32_t i = GradientAccumulator2_InPlaceAccumulatorV2_backward_start; i < GradientAccumulator2_InPlaceAccumulatorV2_backward_stop; i++) {
      DeeployNetwork_input_5[i] = DeeployNetwork_fc2_weight_grad_tensor[i];
      DeeployNetwork_output_2[i] = DeeployNetwork_fc2_weight_grad_tensor[i];
    }
  } else {
    for (int32_t i = GradientAccumulator2_InPlaceAccumulatorV2_backward_start; i < GradientAccumulator2_InPlaceAccumulatorV2_backward_stop; i++) {
      DeeployNetwork_input_5[i] += DeeployNetwork_fc2_weight_grad_tensor[i];
      DeeployNetwork_output_2[i] = DeeployNetwork_input_5[i];
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args_t;

static void _GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork(void *_GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args_t *args =
      (_GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args_t *)_GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args;

  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args_t
      DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args =
          (_GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args_t){.DeeployNetwork_fc2_weight_grad_tensor =
                                                                                          DeeployNetwork_fc2_weight_grad_tensor};

  // _GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure CLOSURE CALL
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure(&DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args_t;

static void _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure(void *_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args_t *args =
      (_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args_t *)_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args;

  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args_t DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args =
      (_GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args_t){.DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork,
                  &DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
} _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args_t;

static void _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3(void *_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args_t *args =
      (_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args_t *)_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args;

  float32_t *DeeployNetwork_fc2_weight_grad_tensor = args->DeeployNetwork_fc2_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args_t DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args =
      (_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args_t){.DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor};

  // _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure CLOSURE CALL
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure(&DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args_t;

static void _GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure(void *_GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args_t *args =
      (_GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args_t *)_GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args;

  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL

  // InPlaceAccumulatorV2 - true in-place (Name: GradientAccumulator1_InPlaceAccumulatorV2_backward, Op: InPlaceAccumulatorV2)
  // Writes result to accum_buffer (in-place) and data_out (explicit output).
  // In training, data_out aliases accum_buffer (same or separate allocation).
  // Reset (lazy_reset_grad=1): accum_buffer  = gradient
  // Accum (lazy_reset_grad=0): accum_buffer += gradient
  int8_t GradientAccumulator1_InPlaceAccumulatorV2_backward_core_id = pi_core_id();
  int8_t GradientAccumulator1_InPlaceAccumulatorV2_backward_log2Core = log2(NUM_CORES);
  int32_t GradientAccumulator1_InPlaceAccumulatorV2_backward_chunk =
      (6272 >> GradientAccumulator1_InPlaceAccumulatorV2_backward_log2Core) + ((6272 & (NUM_CORES - 1)) != 0);
  int32_t GradientAccumulator1_InPlaceAccumulatorV2_backward_start =
      MIN(GradientAccumulator1_InPlaceAccumulatorV2_backward_chunk * GradientAccumulator1_InPlaceAccumulatorV2_backward_core_id, (int32_t)6272);
  int32_t GradientAccumulator1_InPlaceAccumulatorV2_backward_stop =
      MIN(GradientAccumulator1_InPlaceAccumulatorV2_backward_start + GradientAccumulator1_InPlaceAccumulatorV2_backward_chunk, (int32_t)6272);

  if (DeeployNetwork_input_6[0]) {
    for (int32_t i = GradientAccumulator1_InPlaceAccumulatorV2_backward_start; i < GradientAccumulator1_InPlaceAccumulatorV2_backward_stop; i++) {
      DeeployNetwork_input_4[i] = DeeployNetwork_fc1_weight_grad_tensor[i];
      DeeployNetwork_output_1[i] = DeeployNetwork_fc1_weight_grad_tensor[i];
    }
  } else {
    for (int32_t i = GradientAccumulator1_InPlaceAccumulatorV2_backward_start; i < GradientAccumulator1_InPlaceAccumulatorV2_backward_stop; i++) {
      DeeployNetwork_input_4[i] += DeeployNetwork_fc1_weight_grad_tensor[i];
      DeeployNetwork_output_1[i] = DeeployNetwork_input_4[i];
    }
  }

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args_t;

static void _GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork(void *_GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args_t *args =
      (_GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args_t *)_GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args;

  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args_t
      DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args =
          (_GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args_t){.DeeployNetwork_fc1_weight_grad_tensor =
                                                                                          DeeployNetwork_fc1_weight_grad_tensor};

  // _GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure CLOSURE CALL
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure(&DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_tiling_closure_args);

  pi_cl_team_barrier();

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args_t;

static void _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure(void *_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args_t *args =
      (_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args_t *)_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args;

  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args_t DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args =
      (_GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args_t){.DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor};

  pi_cl_team_fork(NUM_CORES, (void *)_GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork,
                  &DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_cluster_fork_args);

  // CLOSURE ARG WRITEBACK
}

typedef struct {
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;
} _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args_t;

static void _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3(void *_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args) {
  // CLOSURE ARG CAST
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args_t *args =
      (_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args_t *)_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args;

  float32_t *DeeployNetwork_fc1_weight_grad_tensor = args->DeeployNetwork_fc1_weight_grad_tensor;

  // CLOSURE FUNCTION CALL
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args_t DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args =
      (_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args_t){.DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor};

  // _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure CLOSURE CALL
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure(&DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_args);

  // CLOSURE ARG WRITEBACK
}

void RunTrainingNetwork() {
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_tensor;
  float32_t *DeeployNetwork_output_tensor;
  float32_t *DeeployNetwork_onnxlog_prob3_tensor;
  float32_t *DeeployNetwork_output_grad_tensor;
  float32_t *DeeployNetwork_fc2_weight_grad_tensor;
  float32_t *DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor;
  float32_t *DeeployNetwork_fc1_weight_grad_tensor;

  DeeployNetwork_node_0_fc1_Gemm__0_tensor = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 8);

  _node_1_fc1_Gemm_Gemm_closure_L3_args_t DeeployNetwork__node_1_fc1_Gemm_Gemm_closure_L3_args =
      (_node_1_fc1_Gemm_Gemm_closure_L3_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor};

  // _node_1_fc1_Gemm_Gemm_closure_L3 CLOSURE CALL
  _node_1_fc1_Gemm_Gemm_closure_L3(&DeeployNetwork__node_1_fc1_Gemm_Gemm_closure_L3_args);

  DeeployNetwork_output_tensor = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 10);

  _node_2_fc2_Gemm_Gemm_closure_L3_args_t DeeployNetwork__node_2_fc2_Gemm_Gemm_closure_L3_args = (_node_2_fc2_Gemm_Gemm_closure_L3_args_t){
      .DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor, .DeeployNetwork_output_tensor = DeeployNetwork_output_tensor};

  // _node_2_fc2_Gemm_Gemm_closure_L3 CLOSURE CALL
  _node_2_fc2_Gemm_Gemm_closure_L3(&DeeployNetwork__node_2_fc2_Gemm_Gemm_closure_L3_args);

  DeeployNetwork_onnxlog_prob3_tensor = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 10);

  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args_t DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args =
      (_onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args_t){.DeeployNetwork_output_tensor = DeeployNetwork_output_tensor,
                                                                                .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor};

  // _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3 CLOSURE CALL
  _onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3(&DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_SoftmaxCrossEntropyLoss_closure_L3_args);

  pi_l2_free(DeeployNetwork_output_tensor, sizeof(float32_t) * 10);

  DeeployNetwork_output_grad_tensor = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 10);

  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args_t
      DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args =
          (_onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args_t){
              .DeeployNetwork_onnxlog_prob3_tensor = DeeployNetwork_onnxlog_prob3_tensor,
              .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor};

  // _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3 CLOSURE CALL
  _onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3(
      &DeeployNetwork__onnxSoftmaxCrossEntropyLoss4_GradSoftmaxCrossEntropyLossGrad_0_SoftmaxCrossEntropyLossGrad_backward_closure_L3_args);

  pi_l2_free(DeeployNetwork_onnxlog_prob3_tensor, sizeof(float32_t) * 10);

  DeeployNetwork_fc2_weight_grad_tensor = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 80);

  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args =
      (_node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_tensor = DeeployNetwork_node_0_fc1_Gemm__0_tensor,
                                                                    .DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                    .DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor};

  // _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3 CLOSURE CALL
  _node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3(&DeeployNetwork__node_2_fc2_Gemm_GradGemm_1_Gemm_backward_closure_L3_args);

  pi_l2_free(DeeployNetwork_node_0_fc1_Gemm__0_tensor, sizeof(float32_t) * 8);

  DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 8);

  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args =
      (_node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t){.DeeployNetwork_output_grad_tensor = DeeployNetwork_output_grad_tensor,
                                                                    .DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor =
                                                                        DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor};

  // _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3 CLOSURE CALL
  _node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3(&DeeployNetwork__node_2_fc2_Gemm_GradGemm_0_Gemm_backward_closure_L3_args);

  pi_l2_free(DeeployNetwork_output_grad_tensor, sizeof(float32_t) * 10);

  DeeployNetwork_fc1_weight_grad_tensor = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 6272);

  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args =
      (_node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args_t){.DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor =
                                                                        DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor,
                                                                    .DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor};

  // _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3 CLOSURE CALL
  _node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3(&DeeployNetwork__node_1_fc1_Gemm_GradGemm_0_Gemm_backward_closure_L3_args);

  pi_l2_free(DeeployNetwork_node_0_fc1_Gemm__0_grad_tensor, sizeof(float32_t) * 8);
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args_t DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args =
      (_GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args_t){.DeeployNetwork_fc2_weight_grad_tensor = DeeployNetwork_fc2_weight_grad_tensor};

  // _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3 CLOSURE CALL
  _GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3(&DeeployNetwork__GradientAccumulator2_InPlaceAccumulatorV2_backward_closure_L3_args);

  pi_l2_free(DeeployNetwork_fc2_weight_grad_tensor, sizeof(float32_t) * 80);
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args_t DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args =
      (_GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args_t){.DeeployNetwork_fc1_weight_grad_tensor = DeeployNetwork_fc1_weight_grad_tensor};

  // _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3 CLOSURE CALL
  _GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3(&DeeployNetwork__GradientAccumulator1_InPlaceAccumulatorV2_backward_closure_L3_args);

  pi_l2_free(DeeployNetwork_fc1_weight_grad_tensor, sizeof(float32_t) * 6272);
}

void InitTrainingNetwork() {

  DeeployNetwork_input_0 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 784);

  DeeployNetwork_input_1 = (uint8_t *)pi_l2_malloc(sizeof(uint8_t) * 1);

  DeeployNetwork_input_2 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 6272);

  DeeployNetwork_input_3 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 80);

  DeeployNetwork_input_4 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 6272);

  DeeployNetwork_input_5 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 80);

  DeeployNetwork_input_6 = (uint8_t *)pi_l2_malloc(sizeof(uint8_t) * 1);

  DeeployNetwork_output_0 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 1);

  DeeployNetwork_output_1 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 6272);

  DeeployNetwork_output_2 = (float32_t *)pi_l2_malloc(sizeof(float32_t) * 80);

  DeeployNetwork_inputs[0] = (void *)DeeployNetwork_input_0;
  DeeployNetwork_inputs[1] = (void *)DeeployNetwork_input_1;
  DeeployNetwork_inputs[2] = (void *)DeeployNetwork_input_2;
  DeeployNetwork_inputs[3] = (void *)DeeployNetwork_input_3;
  DeeployNetwork_inputs[4] = (void *)DeeployNetwork_input_4;
  DeeployNetwork_inputs[5] = (void *)DeeployNetwork_input_5;
  DeeployNetwork_inputs[6] = (void *)DeeployNetwork_input_6;
  DeeployNetwork_outputs[0] = (void *)DeeployNetwork_output_0;
  DeeployNetwork_outputs[1] = (void *)DeeployNetwork_output_1;
  DeeployNetwork_outputs[2] = (void *)DeeployNetwork_output_2;
}
