
/*
 * SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "DeeployPULPMath.h"
#include "pmsis.h"

// ============================================================================
// Minimal pulp-trainlib interface - avoiding pulp_train_defines.h conflicts
// ============================================================================

struct blob {
  float *data;
  float *diff;
  int dim;
  int W;
  int H;
  int C;
};

struct Conv2D_args {
  struct blob *input;
  struct blob *coeff;
  struct blob *bias;
  struct blob *output;
  int Lpad;
  int Rpad;
  int Upad;
  int Dpad;
  int stride_h;
  int stride_w;
  float *i2c_buffer;
  float *bt_buffer;
  int skip_wg_grad;
  int skip_in_grad;
  int HWC;
  int opt_matmul_type_fw;
  int opt_matmul_type_wg;
  int opt_matmul_type_ig;
  int USE_BIASES;
  int USE_IM2COL;
  int USE_DMA_IM2COL;
};

void pulp_conv2d_fp32_bw_param_grads_cl(void *Conv2D_args);
void pulp_conv2d_fp32_bw_input_grads_cl(void *Conv2D_args);

void pulp_conv_dw_fp32_bw_input_grads_cl(void *DepthWise_Conv_args);
void pulp_conv_dw_fp32_bw_param_grads_cl(void *DepthWise_Conv_args);

struct DepthWise_Conv_args {
  struct blob *input;
  struct blob *coeff;
  struct blob *output;

  int stride_h;
  int stride_w;

  int Lpad;
  int Rpad;
  int Upad;
  int Dpad;

  int skip_wg_grad;
  int skip_in_grad;

  int HWC;
};

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
  memset(pGradWeight, 0, sizeof(float) * (C_out * C_in * P * Q));

  struct blob input_blob = {0};
  struct blob output_blob = {0};
  struct blob coeff_blob = {0};
  struct blob bias_blob = {0};

  input_blob.data = (float *)pInput;
  input_blob.diff = NULL;
  input_blob.W = W_in;
  input_blob.H = H_in;
  input_blob.C = C_in;
  input_blob.dim = C_in * H_in * W_in;

  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = W_out;
  output_blob.H = H_out;
  output_blob.C = C_out;
  output_blob.dim = C_out * H_out * W_out;

  coeff_blob.data = NULL;
  coeff_blob.diff = (float *)pGradWeight;
  coeff_blob.W = Q;
  coeff_blob.H = P;
  coeff_blob.C = C_out;
  coeff_blob.dim = C_out * C_in * P * Q;

  bias_blob.data = NULL;
  bias_blob.diff = NULL;
  bias_blob.W = 1;
  bias_blob.H = 1;
  bias_blob.C = C_out;
  bias_blob.dim = C_out;

  struct Conv2D_args conv_args;
  memset(&conv_args, 0, sizeof(conv_args));

  conv_args.input = &input_blob;
  conv_args.output = &output_blob;
  conv_args.coeff = &coeff_blob;
  conv_args.bias = &bias_blob;

  conv_args.Lpad = (int)pad_left;
  conv_args.Rpad = (int)pad_right;
  conv_args.Upad = (int)pad_top;
  conv_args.Dpad = (int)pad_bottom;
  conv_args.stride_h = (int)SP;
  conv_args.stride_w = (int)SQ;

  conv_args.i2c_buffer = NULL;
  conv_args.bt_buffer = NULL;

  conv_args.skip_wg_grad = 0;
  conv_args.skip_in_grad = 1;
  conv_args.HWC = 0;
  conv_args.USE_BIASES = 0;
  conv_args.USE_IM2COL = 0;
  conv_args.USE_DMA_IM2COL = 0;

  pulp_conv2d_fp32_bw_param_grads_cl(&conv_args);
}

void PULP_ConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
  memset(pGradIn, 0, sizeof(float) * (C_in * H_in * W_in));

  struct blob input_blob = (struct blob){0};
  struct blob output_blob = (struct blob){0};
  struct blob coeff_blob = (struct blob){0};
  struct blob bias_blob = (struct blob){0};

  input_blob.data = NULL;
  input_blob.diff = (float *)pGradIn;
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_in;
  input_blob.dim = (int)(C_in * H_in * W_in);

  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_out;
  output_blob.dim = (int)(C_out * H_out * W_out);

  coeff_blob.data = (float *)pWeight;
  coeff_blob.diff = NULL;
  coeff_blob.W = (int)Q;
  coeff_blob.H = (int)P;
  coeff_blob.C = (int)C_out;
  coeff_blob.dim = (int)(C_out * C_in * P * Q);

  bias_blob.data = NULL;
  bias_blob.diff = NULL;
  bias_blob.W = 1;
  bias_blob.H = 1;
  bias_blob.C = (int)C_out;
  bias_blob.dim = (int)C_out;

  struct Conv2D_args conv_args;
  memset(&conv_args, 0, sizeof(conv_args));

  conv_args.input = &input_blob;
  conv_args.output = &output_blob;
  conv_args.coeff = &coeff_blob;
  conv_args.bias = &bias_blob;

  conv_args.Lpad = (int)pad_left;
  conv_args.Rpad = (int)pad_right;
  conv_args.Upad = (int)pad_top;
  conv_args.Dpad = (int)pad_bottom;
  conv_args.stride_h = (int)SP;
  conv_args.stride_w = (int)SQ;

  conv_args.i2c_buffer = NULL;
  conv_args.bt_buffer = NULL;

  conv_args.skip_wg_grad = 1;
  conv_args.skip_in_grad = 0;
  conv_args.HWC = 0;
  conv_args.USE_BIASES = 0;
  conv_args.USE_IM2COL = 0;
  conv_args.USE_DMA_IM2COL = 0;

  pulp_conv2d_fp32_bw_input_grads_cl(&conv_args);
}

void PULP_DWConvTrans2d_fp32_fp32_fp32_HWC(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_total, const float *__restrict__ pWeight, uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ, float *__restrict__ pGradIn, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {

  uint32_t H_in = (H_out - 1) * SP + P - pad_top - pad_bottom;
  uint32_t W_in = (W_out - 1) * SQ + Q - pad_left - pad_right;

  memset(pGradIn, 0, sizeof(float) * (C_total * H_in * W_in));
  struct blob input_blob = {0};
  struct blob coeff_blob = {0};
  struct blob output_blob = {0};

  input_blob.data = NULL;
  input_blob.diff = (float *)pGradIn;
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_total;
  input_blob.dim = (int)(C_total * H_in * W_in);

  coeff_blob.data = (float *)pWeight;
  coeff_blob.diff = NULL;
  coeff_blob.W = (int)Q;
  coeff_blob.H = (int)P;
  coeff_blob.C = (int)C_total;
  coeff_blob.dim = (int)(C_total * P * Q);

  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_total;
  output_blob.dim = (int)(C_total * H_out * W_out);

  struct DepthWise_Conv_args dw_args;
  memset(&dw_args, 0, sizeof(dw_args));

  dw_args.input = &input_blob;
  dw_args.coeff = &coeff_blob;
  dw_args.output = &output_blob;

  dw_args.stride_h = (int)SP;
  dw_args.stride_w = (int)SQ;

  dw_args.Lpad = (int)pad_left;
  dw_args.Rpad = (int)pad_right;
  dw_args.Upad = (int)pad_top;
  dw_args.Dpad = (int)pad_bottom;

  dw_args.skip_wg_grad = 1; 
  dw_args.skip_in_grad = 0; 

  dw_args.HWC = 0; 
  pulp_conv_dw_fp32_bw_input_grads_cl(&dw_args);
}

void PULP_ConvGradB2d_fp32_fp32_NCHW(const float *__restrict__ pGradOut,
                                     uint32_t H_out, uint32_t W_out,
                                     uint32_t C_out,
                                     float *__restrict__ pGradBias) {
  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  uint16_t ch_chunk = (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C_out);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C_out);

  if (ch_start >= ch_stop) {
    return;
  }

  for (uint32_t oc = ch_start; oc < ch_stop; ++oc) {
    float grad_sum = 0.0f;

    for (uint32_t oh = 0; oh < H_out; ++oh) {
      for (uint32_t ow = 0; ow < W_out; ++ow) {
        uint32_t go_idx = (oc * H_out + oh) * W_out + ow;
        grad_sum += pGradOut[go_idx];
      }
  }

    pGradBias[oc] = grad_sum;
  }
}

void PULP_DWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
  // Only supports stride=1
  // No padding support
  // Requires H_out + kernel_size - 1 ≤ H_in

  uint32_t gradw_elems = C_out * (C_in / C_out) * P * Q;

  memset(pGradWeight, 0, sizeof(float) * gradw_elems);

  struct blob input_blob = {0};
  struct blob coeff_blob = {0};
  struct blob output_blob = {0};

  input_blob.data = (float *)pInput;
  input_blob.diff = NULL;
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_in;
  input_blob.dim = (int)(C_in * H_in * W_in);

  coeff_blob.data = NULL;
  coeff_blob.diff = (float *)pGradWeight;
  coeff_blob.W = (int)Q;
  coeff_blob.H = (int)P;
  coeff_blob.C = (int)C_in;
  coeff_blob.dim = (int)(C_in * P * Q);

  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_out;
  output_blob.dim = (int)(C_out * H_out * W_out);

  struct DepthWise_Conv_args dw_args;
  memset(&dw_args, 0, sizeof(dw_args));

  dw_args.input = &input_blob;
  dw_args.coeff = &coeff_blob;
  dw_args.output = &output_blob;

  dw_args.stride_h = (int)SP;
  dw_args.stride_w = (int)SQ;

  dw_args.Lpad = (int)pad_left;
  dw_args.Rpad = (int)pad_right;
  dw_args.Upad = (int)pad_top;
  dw_args.Dpad = (int)pad_bottom;

  dw_args.skip_wg_grad = 0;
  dw_args.skip_in_grad = 1;
  dw_args.HWC = 0; 
  pulp_conv_dw_fp32_bw_param_grads_cl(&dw_args);
}