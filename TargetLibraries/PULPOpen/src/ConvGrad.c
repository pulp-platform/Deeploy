
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

void pulp_conv2d_fp32_bw_param_grads_cl(void *Conv2D_args);
void pulp_conv2d_fp32_bw_input_grads_cl(void *Conv2D_args);

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

void pulp_conv_pw_fp32_bw_param_grads_cl(void *PointWise_Conv_args);
void pulp_conv_pw_fp32_bw_input_grads_cl(void *PointWise_Conv_args);

struct PointWise_Conv_args {
  struct blob *input;
  struct blob *coeff;
  struct blob *output;
  float *transpose_buffer;
  int skip_wg_grad;
  int skip_in_grad;
  int opt_matmul_type_fw;
  int opt_matmul_type_wg;
  int opt_matmul_type_ig;
  int HWC;
};

// Minimal declarations for direct transpose + GEMM in PWConvGradX
struct transp_args {
  float *in_matrix;
  float *out_matrix;
  int N;
  int M;
  int *dim;
  int *transposed_axes;
  int n_dim;
};

struct matMul_args {
  float *__restrict__ A;
  float *__restrict__ B;
  float *__restrict__ C;
  int N;
  int M;
  int K;
  int trans_B;
};

void transpose_matrix(void *void_args);
void mm(void *void_args);

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {

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

void PULP_ConvGradW2d_fp32_fp32_fp32_CHW_Im2Col(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize) {

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
  coeff_blob.C = C_in;
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

  conv_args.i2c_buffer = ctxtBuffer;
  conv_args.bt_buffer = NULL;

  conv_args.skip_wg_grad = 0;
  conv_args.skip_in_grad = 1;
  conv_args.HWC = 0;
  conv_args.USE_BIASES = 0;
  conv_args.USE_IM2COL = 1;
  conv_args.USE_DMA_IM2COL = 0;

  pulp_conv2d_fp32_bw_param_grads_cl(&conv_args);
}

void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_trainlib(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {

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

void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_Im2Col(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right,
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize,
    float *__restrict__ btBuffer, uint32_t btBufferSize) {

  struct blob input_blob = {0};
  struct blob output_blob = {0};
  struct blob coeff_blob = {0};
  struct blob bias_blob = {0};

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

  conv_args.i2c_buffer = ctxtBuffer;
  conv_args.bt_buffer = btBuffer;

  conv_args.skip_wg_grad = 1;
  conv_args.skip_in_grad = 0;
  conv_args.HWC = 0;
  conv_args.USE_BIASES = 0;
  conv_args.USE_IM2COL = 1;
  conv_args.USE_DMA_IM2COL = 0;

  pulp_conv2d_fp32_bw_input_grads_cl(&conv_args);
}

static inline int32_t max_i32(int32_t a, int32_t b) { return (a > b) ? a : b; }
static inline int32_t min_i32(int32_t a, int32_t b) { return (a < b) ? a : b; }

void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, // H_out (tile)
    uint32_t dim_im_out_y, // W_out (tile)
    uint32_t ch_im_out,    // C_out (full)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,     // C_in (full)
    uint32_t dim_kernel_x, // P (kernel H)
    uint32_t dim_kernel_y, // Q (kernel W)
    uint32_t stride_h,     // SH
    uint32_t stride_w,     // SW
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x,      // H_in (tile)
    uint32_t dim_im_in_y,      // W_in (tile)
    uint32_t padding_x_left,   // pad_top
    uint32_t padding_x_right,  // pad_bottom (unused here)
    uint32_t padding_y_top,    // pad_left
    uint32_t padding_y_bottom, // pad_right (unused here)
    uint16_t offset_grad_in_h, uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h, uint16_t offset_grad_out_w) {
  (void)padding_x_right;
  (void)padding_y_bottom;

  const uint32_t Hout_t = dim_im_out_x;
  const uint32_t Wout_t = dim_im_out_y;
  const uint32_t Hin_t = dim_im_in_x;
  const uint32_t Win_t = dim_im_in_y;

  const uint32_t Cout = ch_im_out;
  const uint32_t Cin = ch_im_in;

  const uint32_t P = dim_kernel_x;
  const uint32_t Q = dim_kernel_y;

  const int32_t pad_top = (int32_t)padding_x_left;
  const int32_t pad_left = (int32_t)padding_y_top;

  const int32_t sh = (int32_t)stride_h;
  const int32_t sw = (int32_t)stride_w;

  const int32_t hx0 = (int32_t)offset_grad_in_h;
  const int32_t wx0 = (int32_t)offset_grad_in_w;
  const int32_t hx1 = hx0 + (int32_t)Hin_t - 1;
  const int32_t wx1 = wx0 + (int32_t)Win_t - 1;

  // -------- core partition over Cin --------
  const int core_id = pi_core_id();
  const int ncores = NUM_CORES;

  const uint32_t ci_chunk = (Cin + (uint32_t)ncores - 1u) / (uint32_t)ncores;
  const uint32_t ci_start = (uint32_t)core_id * ci_chunk;
  uint32_t ci_stop = ci_start + ci_chunk;
  if (ci_stop > Cin)
    ci_stop = Cin;

  if (ci_start >= ci_stop) {
    return;
  }

  for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
    float *dx_ci = pGradIn + (size_t)ci * Hin_t * Win_t;
    for (uint32_t ih = 0; ih < Hin_t; ++ih) {
      for (uint32_t iw = 0; iw < Win_t; ++iw) {
        dx_ci[ih * Win_t + iw] = 0.0f;
      }
    }
  }

  for (uint32_t co = 0; co < Cout; ++co) {
    const float *dy_co = pGradOut + (size_t)co * Hout_t * Wout_t;

    for (uint32_t ly = 0; ly < Hout_t; ++ly) {
      const int32_t oy = (int32_t)offset_grad_out_h + (int32_t)ly;
      const int32_t base_h = oy * sh - pad_top;

      for (uint32_t lx = 0; lx < Wout_t; ++lx) {
        const int32_t ox = (int32_t)offset_grad_out_w + (int32_t)lx;
        const int32_t base_w = ox * sw - pad_left;

        const float dy_val = dy_co[ly * Wout_t + lx];

        // prune ky/kx once per (co,ly,lx) (independent of ci)
        int32_t ky_min = max_i32(0, hx0 - base_h);
        int32_t ky_max = min_i32((int32_t)P - 1, hx1 - base_h);
        if (ky_min > ky_max)
          continue;

        int32_t kx_min = max_i32(0, wx0 - base_w);
        int32_t kx_max = min_i32((int32_t)Q - 1, wx1 - base_w);
        if (kx_min > kx_max)
          continue;

        for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
          float *dx_ci = pGradIn + (size_t)ci * Hin_t * Win_t;

          // W[co,ci,:,:] base (assumes layout [Cout][Cin][P][Q])
          const float *w_co_ci =
              pWeight +
              (((size_t)co * (size_t)Cin + (size_t)ci) * (size_t)P * (size_t)Q);

          for (int32_t ky = ky_min; ky <= ky_max; ++ky) {
            const int32_t ih = (base_h + ky) - hx0; // local in tile

            for (int32_t kx = kx_min; kx <= kx_max; ++kx) {
              const int32_t iw = (base_w + kx) - wx0;

              dx_ci[(uint32_t)ih * Win_t + (uint32_t)iw] +=
                  dy_val * w_co_ci[(size_t)ky * (size_t)Q + (size_t)kx];
            }
          }
        }
      }
    }
  }
}

void PULP_ConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, // H_out (tile)
    uint32_t dim_im_out_y, // W_out (tile)
    uint32_t ch_im_out,    // C_out (full)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,     // C_in (full)
    uint32_t dim_kernel_x, // P
    uint32_t dim_kernel_y, // Q
    uint32_t stride_h,     // SP
    uint32_t stride_w,     // SQ
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x,     // H_in (tile)
    uint32_t dim_im_in_y,     // W_in (tile)
    uint32_t padding_x_left,  // pad_top (tile-specific)
    uint32_t padding_x_right, // pad_bottom (tile-specific)
    uint32_t padding_y_top,   // pad_left (tile-specific)
    uint32_t padding_y_bottom // pad_right (tile-specific)
) {
  // Map to more intuitive names
  const uint32_t H_out = dim_im_out_x;
  const uint32_t W_out = dim_im_out_y;
  const uint32_t C_out = ch_im_out;
  const uint32_t C_in = ch_im_in;
  const uint32_t P = dim_kernel_x;
  const uint32_t Q = dim_kernel_y;
  const uint32_t H_in = dim_im_in_x;
  const uint32_t W_in = dim_im_in_y;
  const uint32_t pad_top = padding_x_left;
  const uint32_t pad_bottom = padding_x_right;
  const uint32_t pad_left = padding_y_top;
  const uint32_t pad_right = padding_y_bottom;

  int8_t core_id = pi_core_id();
  int8_t log2Core = LOG2(NUM_CORES);

  // Parallelize over input channels (C_in)
  uint16_t ch_chunk = (C_in >> log2Core) + ((C_in & (NUM_CORES - 1)) != 0);
  uint16_t ch_start = MIN(ch_chunk * core_id, C_in);
  uint16_t ch_stop = MIN(ch_start + ch_chunk, C_in);

  // =========================================================================
  // Step 1: Zero-initialize dX tile for this core's channel range
  // =========================================================================
  // CHW layout: [C, H, W]
  for (uint32_t c_in = ch_start; c_in < ch_stop; ++c_in) {
    for (uint32_t h = 0; h < H_in; ++h) {
      for (uint32_t w = 0; w < W_in; ++w) {
        uint32_t dx_idx = (c_in * H_in + h) * W_in + w;
        pGradIn[dx_idx] = 0.0f;
      }
    }
  }

  // =========================================================================
  // Step 2: Compute gradient via transposed convolution
  // =========================================================================
  // For each input channel assigned to this core
  for (uint32_t c_in = ch_start; c_in < ch_stop; ++c_in) {
    // For each kernel position
    for (uint32_t kh = 0; kh < P; ++kh) {
      for (uint32_t kw = 0; kw < Q; ++kw) {
        // For each output position in dY tile
        for (uint32_t h_out = 0; h_out < H_out; ++h_out) {
          // Compute corresponding input position
          int32_t h_in = (int32_t)h_out * (int32_t)stride_h + (int32_t)kh -
                         (int32_t)pad_top;

          // Check bounds (tile-local)
          if (h_in < 0 || h_in >= (int32_t)H_in) {
            continue;
          }

          for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
            // Compute corresponding input position
            int32_t w_in = (int32_t)w_out * (int32_t)stride_w + (int32_t)kw -
                           (int32_t)pad_left;

            // Check bounds (tile-local)
            if (w_in < 0 || w_in >= (int32_t)W_in) {
              continue;
            }

            // Accumulate gradient contributions from all output channels
            // dX index: CHW layout [C_in, H_in, W_in]
            uint32_t dx_idx =
                (c_in * H_in + (uint32_t)h_in) * W_in + (uint32_t)w_in;

            for (uint32_t c_out = 0; c_out < C_out; ++c_out) {
              // dY index: CHW layout [C_out, H_out, W_out]
              uint32_t dy_idx = (c_out * H_out + h_out) * W_out + w_out;

              // Weight index: [C_out, C_in, P, Q] layout
              uint32_t w_idx = ((c_out * C_in + c_in) * P + kh) * Q + kw;

              // Accumulate: dX += dY * W
              pGradIn[dx_idx] += pGradOut[dy_idx] * pWeight[w_idx];
            }
          }
        }
      }
    }
  }
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

void PULP_DWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
    uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
    uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
  // Supports depthwise convolution with multiplier
  // For depthwise: groups = C_in, C_out = C_in * multiplier
  // Weight shape: [C_out, 1, P, Q]

  uint32_t gradw_elems = C_out * (C_in / C_out) * P * Q;

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
  coeff_blob.C = (int)C_out; // Fixed: should be C_out for DW with multiplier
  coeff_blob.dim = (int)(C_out * P * Q); // Fixed: total weight elements

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

// ============================================================================
// Pointwise Convolution Gradient Functions (using pulptrainlib pw interfaces)
// ============================================================================

void PULP_PWConvGradW2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
    uint32_t W_in, uint32_t C_in, float *__restrict__ pGradWeight) {

  struct blob input_blob = {0};
  struct blob output_blob = {0};
  struct blob coeff_blob = {0};

  // Input blob (forward activation)
  input_blob.data = (float *)pInput;
  input_blob.diff = NULL;
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_in;
  input_blob.dim = (int)(C_in * H_in * W_in);

  // Output blob (gradient w.r.t. output)
  output_blob.data = NULL;
  output_blob.diff = (float *)pGradOut;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_out;
  output_blob.dim = (int)(C_out * H_out * W_out);

  // Weight blob (gradient w.r.t. weights - output)
  // For PW conv: kernel is 1x1, so dim = C_out * C_in
  coeff_blob.data = NULL;
  coeff_blob.diff = (float *)pGradWeight;
  coeff_blob.W = 1;
  coeff_blob.H = 1;
  coeff_blob.C = (int)C_in;
  coeff_blob.dim = (int)(C_out * C_in);

  struct PointWise_Conv_args pw_args;
  memset(&pw_args, 0, sizeof(pw_args));

  pw_args.input = &input_blob;
  pw_args.output = &output_blob;
  pw_args.coeff = &coeff_blob;
  pw_args.transpose_buffer = NULL;

  pw_args.skip_wg_grad = 0; // Compute weight gradient
  pw_args.skip_in_grad = 1; // Skip input gradient
  pw_args.HWC = 0;          // CHW layout
  pw_args.opt_matmul_type_fw = 0;
  pw_args.opt_matmul_type_wg = 0;
  pw_args.opt_matmul_type_ig = 0;

  pulp_conv_pw_fp32_bw_param_grads_cl(&pw_args);
}

// Direct PW ConvGradX worker. Eliminates the per-call weight transpose +
// pulp-trainlib mm: those required a Cin*Cout transient buffer in L1, which
// for MobileNetV1 block 6-10 PW layers (Cin=Cout=128) eats 64 KB of the
// 128 KB scratch and forces the tiler to fragment Cin/H/W into ~36 tiles.
// Each tile then pays an L3<->L2 DMA + sync cost ~25x larger than the
// actual compute. The direct kernel keeps the inner axpy fully contiguous
// and parallelizes over Cin without scratch.
typedef struct {
  const float *pGradOut;
  const float *pWeight;
  float *pGradIn;
  uint32_t C_out;
  uint32_t C_in;
  uint32_t HW;
} pw_convgradx_args_t;

static void pulp_pw_convgradx_fp32_worker(void *arg_) {
  const pw_convgradx_args_t *a = (const pw_convgradx_args_t *)arg_;
  const uint32_t Cin = a->C_in;
  const uint32_t Cout = a->C_out;
  const uint32_t HW = a->HW;
  const float *__restrict__ pGradOut = a->pGradOut;
  const float *__restrict__ pWeight = a->pWeight;
  float *__restrict__ pGradIn = a->pGradIn;

  // Each core owns a contiguous Cin range
  const uint32_t ci_per_core = (Cin + NUM_CORES - 1u) / NUM_CORES;
  const uint32_t ci_lo = (uint32_t)pi_core_id() * ci_per_core;
  uint32_t ci_hi = ci_lo + ci_per_core;
  if (ci_hi > Cin)
    ci_hi = Cin;
  if (ci_lo >= ci_hi)
    return;

  // Zero this core's slice of dX
  for (uint32_t ci = ci_lo; ci < ci_hi; ++ci) {
    float *dx_row = pGradIn + (size_t)ci * HW;
    for (uint32_t hw = 0; hw < HW; ++hw)
      dx_row[hw] = 0.0f;
  }

  // dX[ci, hw] = sum_co W[co, ci] * dY[co, hw]
  // Outer co loop streams a contiguous W row (length Cin) and a contiguous
  // dY row (length HW); inner axpy over hw stays contiguous on dX.
  for (uint32_t co = 0; co < Cout; ++co) {
    const float *__restrict__ w_row = pWeight + (size_t)co * Cin;
    const float *__restrict__ dy_row = pGradOut + (size_t)co * HW;
    for (uint32_t ci = ci_lo; ci < ci_hi; ++ci) {
      const float w = w_row[ci];
      float *__restrict__ dx_row = pGradIn + (size_t)ci * HW;
      for (uint32_t hw = 0; hw < HW; ++hw)
        dx_row[hw] += w * dy_row[hw];
    }
  }
}

void PULP_PWConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in) {

  // PW (1x1) has H_in == H_out and W_in == W_out at stride=1, pad=0.
  (void)H_in;
  (void)W_in;

  pw_convgradx_args_t args = {
      .pGradOut = pGradOut,
      .pWeight = pWeight,
      .pGradIn = pGradIn,
      .C_out = C_out,
      .C_in = C_in,
      .HW = H_out * W_out,
  };
  pi_cl_team_fork(NUM_CORES, pulp_pw_convgradx_fp32_worker, &args);
}

// Tile-aware Im2Col-based ConvGradX kernel with offset support
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_Im2Col_tiled(
    const float *__restrict__ pGradOut, // dY tile (L1)
    uint32_t dim_im_out_x,              // dY tile H
    uint32_t dim_im_out_y,              // dY tile W
    uint32_t ch_im_out,                 // C_out (full)
    const float *__restrict__ pWeight,  // W
    uint32_t ch_im_in,                  // C_in (full)
    uint32_t dim_kernel_x,              // P (kernel H)
    uint32_t dim_kernel_y,              // Q (kernel W)
    uint32_t stride_h,                  // stride H
    uint32_t stride_w,                  // stride W
    float *__restrict__ pGradIn,        // dX tile (L1)
    uint32_t dim_im_in_x,               // dX tile H
    uint32_t dim_im_in_y,               // dX tile W
    uint32_t padding_y_top,             // pad top (tile-specific)
    uint32_t padding_y_bottom,          // pad bottom (tile-specific)
    uint32_t padding_x_left,            // pad left (tile-specific)
    uint32_t padding_x_right,           // pad right (tile-specific)
    uint16_t offset_grad_in_h,          // dX tile offset H (global)
    uint16_t offset_grad_in_w,          // dX tile offset W (global)
    uint16_t offset_grad_out_h,         // dY tile offset H (global)
    uint16_t offset_grad_out_w,         // dY tile offset W (global)
    float *__restrict__ ctxtBuffer, uint32_t ctxtBufferSize,
    float *__restrict__ btBuffer, uint32_t btBufferSize) {
  const uint32_t Hout_t = dim_im_out_x;
  const uint32_t Wout_t = dim_im_out_y;
  const uint32_t Hin_t = dim_im_in_x;
  const uint32_t Win_t = dim_im_in_y;

  const uint32_t Cout = ch_im_out;
  const uint32_t Cin = ch_im_in;

  const uint32_t P = dim_kernel_x;
  const uint32_t Q = dim_kernel_y;

  // ── im2col+GEMM with internal Cout blocking (supports any stride) ──
  //
  // Math: dX[ci,h,w] = sum_co sum_ky sum_kx dY[co, oy, ox] * W[co,ci,ky,kx]
  //   where oy = (h+pad-ky)/stride, ox = (w+pad-kx)/stride
  //   only when (h+pad-ky) % stride == 0 and (w+pad-kx) % stride == 0
  //
  // Reformulated as GEMM per Cout block:
  //   1. Build dY_col[co_size * P * Q,  Hin * Win] in ctxtBuffer
  //   2. Transpose W[co_size, Cin, P, Q] → W_flat[Cin, co_size * P * Q] in
  //   btBuffer
  //   3. GEMM: dX[Cin, Hin*Win] += W_flat × dY_col
  //
  // Buffer sizes:
  //   ctxtBuffer: co_block * P * Q * Hin * Win * sizeof(float)
  //   btBuffer:   Cin * co_block * P * Q * sizeof(float)
  //
  if (stride_h == 1 && stride_w == 1 && ctxtBuffer != NULL &&
      btBuffer != NULL) {
    uint32_t co_block = Cout;
    while (co_block > 1) {
      uint32_t i2c_need =
          co_block * P * Q * Hin_t * Win_t * (uint32_t)sizeof(float);
      uint32_t bt_need = Cin * co_block * P * Q * (uint32_t)sizeof(float);
      if (i2c_need <= ctxtBufferSize && bt_need <= btBufferSize)
        break;
      co_block /= 2;
    }
    uint32_t i2c_need =
        co_block * P * Q * Hin_t * Win_t * (uint32_t)sizeof(float);
    uint32_t bt_need = Cin * co_block * P * Q * (uint32_t)sizeof(float);

    if (i2c_need <= ctxtBufferSize && bt_need <= btBufferSize) {
      // Zero dX once (core 0 only, then barrier)
      if (pi_core_id() == 0) {
        memset(pGradIn, 0, Cin * Hin_t * Win_t * sizeof(float));
      }
      pi_cl_team_barrier(0);

      const int32_t pad_t = (int32_t)padding_y_top;
      const int32_t pad_l = (int32_t)padding_x_left;

      for (uint32_t co_start = 0; co_start < Cout; co_start += co_block) {
        uint32_t co_size =
            (co_start + co_block > Cout) ? (Cout - co_start) : co_block;
        const float *dy_blk = pGradOut + (size_t)co_start * Hout_t * Wout_t;
        const float *w_blk = pWeight + (size_t)co_start * Cin * P * Q;

        // ── Step 1: Build dY_col in ctxtBuffer (all cores) ──
        // Layout: dY_col[row, col] where row = co*P*Q + ky*Q + kx, col =
        // h*Win+w dY_col[co*P*Q + ky*Q + kx, h*Win + w] = dY[co_start+co,
        // h+pad-ky, w+pad-kx]
        //   (0 if out of bounds)
        {
          uint32_t total_rows = co_size * P * Q;
          uint32_t total_cols = Hin_t * Win_t;
          // Fill dY_col: row = co*P*Q + ky*Q + kx, col = h*Win + w
          // dY_col[row, col] = dY[co, h+pad-ky, w+pad-kx]  (0 if OOB)
          // Parallel over co
          {
            uint32_t co_chunk = (co_size + NUM_CORES - 1) / NUM_CORES;
            uint32_t co_lo = pi_core_id() * co_chunk;
            uint32_t co_hi =
                co_lo + co_chunk > co_size ? co_size : co_lo + co_chunk;
            for (uint32_t co = co_lo; co < co_hi; ++co) {
              const float *dy_co = dy_blk + (size_t)co * Hout_t * Wout_t;
              for (uint32_t ky = 0; ky < P; ++ky) {
                for (uint32_t kx = 0; kx < Q; ++kx) {
                  uint32_t row = co * P * Q + ky * Q + kx;
                  float *dst_row = ctxtBuffer + (size_t)row * total_cols;
                  for (uint32_t h = 0; h < Hin_t; ++h) {
                    int32_t h_off = (int32_t)h + pad_t - (int32_t)ky;
                    for (uint32_t w = 0; w < Win_t; ++w) {
                      int32_t w_off = (int32_t)w + pad_l - (int32_t)kx;
                      float val = 0.0f;
                      // stride check: h_off and w_off must be divisible by
                      // stride
                      if (h_off >= 0 && w_off >= 0 &&
                          (h_off % (int32_t)stride_h) == 0 &&
                          (w_off % (int32_t)stride_w) == 0) {
                        int32_t oy = h_off / (int32_t)stride_h;
                        int32_t ox = w_off / (int32_t)stride_w;
                        if (oy < (int32_t)Hout_t && ox < (int32_t)Wout_t) {
                          val = dy_co[(uint32_t)oy * Wout_t + (uint32_t)ox];
                        }
                      }
                      dst_row[h * Win_t + w] = val;
                    }
                  }
                }
              }
            }
          } // end parallel im2col
        }
        pi_cl_team_barrier(0);

        // ── Step 2: Transpose W into btBuffer (parallel over Cin) ──
        // W_flat[ci, co*P*Q + ky*Q + kx] = W_blk[co, ci, ky, kx]
        {
          uint32_t KPQ = co_size * P * Q;
          uint32_t ci_chunk2 = (Cin + NUM_CORES - 1) / NUM_CORES;
          uint32_t ci_start = pi_core_id() * ci_chunk2;
          uint32_t ci_end =
              ci_start + ci_chunk2 > Cin ? Cin : ci_start + ci_chunk2;

          for (uint32_t ci = ci_start; ci < ci_end; ++ci) {
            float *dst_ci = btBuffer + (size_t)ci * KPQ;
            for (uint32_t co = 0; co < co_size; ++co) {
              const float *src = w_blk + ((size_t)co * Cin + ci) * P * Q;
              float *dst = dst_ci + (size_t)co * P * Q;
              for (uint32_t i = 0; i < P * Q; ++i) {
                dst[i] = src[i];
              }
            }
          }
        }
        pi_cl_team_barrier(0);

        // ── Step 3: GEMM  dX += W_flat × dY_col (parallel over Cin) ──
        {
          uint32_t K = co_size * P * Q;
          uint32_t M = Hin_t * Win_t;
          uint32_t ci_chunk3 = (Cin + NUM_CORES - 1) / NUM_CORES;
          uint32_t ci_start = pi_core_id() * ci_chunk3;
          uint32_t ci_end =
              ci_start + ci_chunk3 > Cin ? Cin : ci_start + ci_chunk3;

          for (uint32_t ci = ci_start; ci < ci_end; ++ci) {
            const float *a_row = btBuffer + (size_t)ci * K;
            float *c_row = pGradIn + (size_t)ci * M;
            // Process 4 output columns at a time for better B-row locality
            uint32_t m = 0;
            for (; m + 3 < M; m += 4) {
              float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
              for (uint32_t k = 0; k < K; ++k) {
                float a_val = a_row[k];
                const float *b_row = ctxtBuffer + k * M + m;
                acc0 += a_val * b_row[0];
                acc1 += a_val * b_row[1];
                acc2 += a_val * b_row[2];
                acc3 += a_val * b_row[3];
              }
              c_row[m] += acc0;
              c_row[m + 1] += acc1;
              c_row[m + 2] += acc2;
              c_row[m + 3] += acc3;
            }
            for (; m < M; ++m) {
              float acc = 0.0f;
              for (uint32_t k = 0; k < K; ++k) {
                acc += a_row[k] * ctxtBuffer[k * M + m];
              }
              c_row[m] += acc;
            }
          }
        }
        pi_cl_team_barrier(0);
      }
      return;
    }
  }

  // ── Fallback: direct 7-deep loop (stride>1 or buffers too small) ──
  {
    const int32_t pad_top = (int32_t)padding_y_top;
    const int32_t pad_left = (int32_t)padding_x_left;
    const int32_t sh = (int32_t)stride_h;
    const int32_t sw = (int32_t)stride_w;
    const int32_t hx0 = (int32_t)offset_grad_in_h;
    const int32_t wx0 = (int32_t)offset_grad_in_w;
    const int32_t hx1 = hx0 + (int32_t)Hin_t - 1;
    const int32_t wx1 = wx0 + (int32_t)Win_t - 1;

    const int core_id = pi_core_id();
    const uint32_t ci_chunk = (Cin + NUM_CORES - 1u) / NUM_CORES;
    const uint32_t ci_start = (uint32_t)core_id * ci_chunk;
    uint32_t ci_stop = ci_start + ci_chunk;
    if (ci_stop > Cin)
      ci_stop = Cin;
    if (ci_start >= ci_stop)
      return;

    for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
      float *dx_ci = pGradIn + (size_t)ci * Hin_t * Win_t;
      for (uint32_t i = 0; i < Hin_t * Win_t; ++i)
        dx_ci[i] = 0.0f;
    }

    // Loop reorder: ci outermost (parallel), then ky/kx, then co (innermost)
    // → W[co, ci, ky, kx] access with co varying fastest = stride-1 sequential
    // → dY[co, ly, lx] also sequential in co
    for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
      float *dx_ci = pGradIn + (size_t)ci * Hin_t * Win_t;
      for (int32_t ky = 0; ky < (int32_t)P; ++ky) {
        for (int32_t kx = 0; kx < (int32_t)Q; ++kx) {
          for (uint32_t ly = 0; ly < Hout_t; ++ly) {
            const int32_t base_h =
                ((int32_t)offset_grad_out_h + (int32_t)ly) * sh - pad_top;
            const int32_t ih = base_h + ky - hx0;
            if (ih < 0 || ih >= (int32_t)Hin_t)
              continue;
            for (uint32_t lx = 0; lx < Wout_t; ++lx) {
              const int32_t base_w =
                  ((int32_t)offset_grad_out_w + (int32_t)lx) * sw - pad_left;
              const int32_t iw = base_w + kx - wx0;
              if (iw < 0 || iw >= (int32_t)Win_t)
                continue;
              // Inner loop over co: W and dY accessed sequentially
              float acc = 0.0f;
              for (uint32_t co = 0; co < Cout; ++co) {
                acc += pGradOut[co * Hout_t * Wout_t + ly * Wout_t + lx] *
                       pWeight[(co * Cin + ci) * P * Q + ky * (int32_t)Q + kx];
              }
              dx_ci[(uint32_t)ih * Win_t + (uint32_t)iw] += acc;
            }
          }
        }
      }
    }
  }
}

void PULP_DWConvGradX2d_fp32_fp32_fp32_CHW_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x, // H_out (tile)
    uint32_t dim_im_out_y, // W_out (tile)
    uint32_t ch_im_out,    // C_out (full)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,     // C_in (full)
    uint32_t dim_kernel_x, // P (kernel H)
    uint32_t dim_kernel_y, // Q (kernel W)
    uint32_t stride_h,     // SH
    uint32_t stride_w,     // SW
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x,      // H_in (tile)
    uint32_t dim_im_in_y,      // W_in (tile)
    uint32_t padding_x_left,   // pad_top
    uint32_t padding_x_right,  // pad_bottom (unused here)
    uint32_t padding_y_top,    // pad_left
    uint32_t padding_y_bottom, // pad_right (unused here)
    uint16_t offset_grad_in_h, uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h, uint16_t offset_grad_out_w) {
  (void)padding_x_right;
  (void)padding_y_bottom;

  const uint32_t Hout_t = dim_im_out_x;
  const uint32_t Wout_t = dim_im_out_y;
  const uint32_t Hin_t = dim_im_in_x;
  const uint32_t Win_t = dim_im_in_y;

  const uint32_t Cout_full = ch_im_out;
  const uint32_t Cin_full = ch_im_in;

  const uint32_t P = dim_kernel_x;
  const uint32_t Q = dim_kernel_y;

  const int32_t pad_top = (int32_t)padding_x_left;
  const int32_t pad_left = (int32_t)padding_y_top;

  const int32_t sh = (int32_t)stride_h;
  const int32_t sw = (int32_t)stride_w;

  // dx tile global box [hx0..hx1] x [wx0..wx1]
  const int32_t hx0 = (int32_t)offset_grad_in_h;
  const int32_t wx0 = (int32_t)offset_grad_in_w;
  const int32_t hx1 = hx0 + (int32_t)Hin_t - 1;
  const int32_t wx1 = wx0 + (int32_t)Win_t - 1;

  // -------- Grouped/Depthwise Convolution Parameters --------
  // For depthwise: groups = Cin, channels_per_group_in = 1
  // For grouped: groups divides both Cin and Cout
  // Assume groups = Cin (standard depthwise with multiplier)
  const uint32_t groups = Cin_full;
  const uint32_t channels_per_group_out = Cout_full / groups;

  // -------- core partition over input channels --------
  const int core_id = pi_core_id();
  const int ncores = NUM_CORES;

  const uint32_t ci_chunk =
      (Cin_full + (uint32_t)ncores - 1u) / (uint32_t)ncores;
  const uint32_t ci_start = (uint32_t)core_id * ci_chunk;
  uint32_t ci_stop = ci_start + ci_chunk;
  if (ci_stop > Cin_full)
    ci_stop = Cin_full;

  if (ci_start >= ci_stop) {
    return;
  }

  // ---- Clear dx tile for this core's input channels ----
  for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
    float *dx_ci = pGradIn + (size_t)ci * (size_t)Hin_t * (size_t)Win_t;

    for (uint32_t ih = 0; ih < Hin_t; ++ih) {
      float *row = dx_ci + (size_t)ih * (size_t)Win_t;
      for (uint32_t iw = 0; iw < Win_t; ++iw) {
        row[iw] = 0.0f;
      }
    }
  }

  // ---- Main computation: scatter from dy to dx ----
  // For each input channel assigned to this core
  for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
    float *dx_ci = pGradIn + (size_t)ci * (size_t)Hin_t * (size_t)Win_t;

    // Determine which output channels contribute to this input channel
    // For depthwise with multiplier: input channel ci corresponds to
    // output channels [ci * channels_per_group_out, (ci+1) *
    // channels_per_group_out)
    const uint32_t co_start = ci * channels_per_group_out;
    const uint32_t co_stop = co_start + channels_per_group_out;

    // Accumulate gradients from all corresponding output channels
    for (uint32_t co = co_start; co < co_stop; ++co) {
      const float *dy_co =
          pGradOut + (size_t)co * (size_t)Hout_t * (size_t)Wout_t;

      // DW weight layout: [Cout][1][P][Q] -> for channel co, weights at
      // [co][P][Q]
      const float *w_co = pWeight + (size_t)co * (size_t)P * (size_t)Q;

      // ---- Scatter from dy tile into dx tile ----
      for (uint32_t ly = 0; ly < Hout_t; ++ly) {
        const int32_t oy = (int32_t)offset_grad_out_h + (int32_t)ly;
        const int32_t base_h = oy * sh - pad_top;

        for (uint32_t lx = 0; lx < Wout_t; ++lx) {
          const int32_t ox = (int32_t)offset_grad_out_w + (int32_t)lx;
          const int32_t base_w = ox * sw - pad_left;

          const float dy_val = dy_co[ly * Wout_t + lx];

          // Intersect kernel footprint with dx tile bounds
          int32_t ky_min = max_i32(0, hx0 - base_h);
          int32_t ky_max = min_i32((int32_t)P - 1, hx1 - base_h);
          if (ky_min > ky_max)
            continue;

          int32_t kx_min = max_i32(0, wx0 - base_w);
          int32_t kx_max = min_i32((int32_t)Q - 1, wx1 - base_w);
          if (kx_min > kx_max)
            continue;

          for (int32_t ky = ky_min; ky <= ky_max; ++ky) {
            const int32_t ih = (base_h + ky) - hx0; // local in dx tile

            for (int32_t kx = kx_min; kx <= kx_max; ++kx) {
              const int32_t iw = (base_w + kx) - wx0;

              const size_t w_idx =
                  (size_t)(uint32_t)ky * (size_t)Q + (size_t)(uint32_t)kx;

              dx_ci[(size_t)(uint32_t)ih * (size_t)Win_t +
                    (size_t)(uint32_t)iw] += dy_val * w_co[w_idx];
            }
          }
        }
      }
    }
  }
}
