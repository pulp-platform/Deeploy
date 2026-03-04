
#ifndef __DEEPLOY_TRAINING_HEADER__
#define __DEEPLOY_TRAINING_HEADER__
#include "DeeployPULPMath.h"
#include "bsp/ram.h"
#include "dory_mem.h"
#include "mchan_siracusa.h"
#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "stdint.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
void RunTrainingNetwork();
void InitTrainingNetwork();

extern float32_t *DeeployNetwork_input_0;
static const uint32_t DeeployNetwork_input_0_len = 784;
extern uint8_t *DeeployNetwork_input_1;
static const uint32_t DeeployNetwork_input_1_len = 1;
extern float32_t *DeeployNetwork_input_2;
static const uint32_t DeeployNetwork_input_2_len = 6272;
extern float32_t *DeeployNetwork_input_3;
static const uint32_t DeeployNetwork_input_3_len = 80;
extern float32_t *DeeployNetwork_input_4;
static const uint32_t DeeployNetwork_input_4_len = 6272;
extern float32_t *DeeployNetwork_input_5;
static const uint32_t DeeployNetwork_input_5_len = 80;
extern uint8_t *DeeployNetwork_input_6;
static const uint32_t DeeployNetwork_input_6_len = 1;
extern float32_t *DeeployNetwork_output_0;
static const uint32_t DeeployNetwork_output_0_len = 1.0;
extern float32_t *DeeployNetwork_output_1;
static const uint32_t DeeployNetwork_output_1_len = 6272;
extern float32_t *DeeployNetwork_output_2;
static const uint32_t DeeployNetwork_output_2_len = 80;
static const uint32_t DeeployNetwork_num_inputs = 7;
static const uint32_t DeeployNetwork_num_outputs = 3;
extern void *DeeployNetwork_inputs[7];
extern void *DeeployNetwork_outputs[3];
static const uint32_t DeeployNetwork_inputs_bytes[7] = {3136, 1, 25088, 320, 25088, 320, 1};
static const uint32_t DeeployNetwork_outputs_bytes[3] = {4.0, 25088, 320};
#endif
