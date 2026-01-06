
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
    uint32_t dim_im_out_x,    // H_out (tile)
    uint32_t dim_im_out_y,    // W_out (tile)
    uint32_t ch_im_out,       // C_out (full)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,        // C_in (full)
    uint32_t dim_kernel_x,    // P (kernel H)
    uint32_t dim_kernel_y,    // Q (kernel W)
    uint32_t stride_h,        // SH
    uint32_t stride_w,        // SW
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x,     // H_in (tile)
    uint32_t dim_im_in_y,     // W_in (tile)
    uint32_t padding_x_left,  // pad_top
    uint32_t padding_x_right, // pad_bottom (unused here)
    uint32_t padding_y_top,   // pad_left
    uint32_t padding_y_bottom,// pad_right (unused here)
    uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w
){
    (void)padding_x_right;
    (void)padding_y_bottom;

    const uint32_t Hout_t = dim_im_out_x;
    const uint32_t Wout_t = dim_im_out_y;
    const uint32_t Hin_t  = dim_im_in_x;
    const uint32_t Win_t  = dim_im_in_y;

    const uint32_t Cout = ch_im_out;
    const uint32_t Cin  = ch_im_in;

    const uint32_t P = dim_kernel_x;
    const uint32_t Q = dim_kernel_y;

    const int32_t pad_top  = (int32_t)padding_x_left;
    const int32_t pad_left = (int32_t)padding_y_top;

    const int32_t sh = (int32_t)stride_h;
    const int32_t sw = (int32_t)stride_w;

    const int32_t hx0 = (int32_t)offset_grad_in_h;
    const int32_t wx0 = (int32_t)offset_grad_in_w;
    const int32_t hx1 = hx0 + (int32_t)Hin_t - 1;
    const int32_t wx1 = wx0 + (int32_t)Win_t - 1;

    // -------- core partition over Cin --------
    const int core_id = pi_core_id();
    const int ncores  = NUM_CORES;

    const uint32_t ci_chunk = (Cin + (uint32_t)ncores - 1u) / (uint32_t)ncores;
    const uint32_t ci_start = (uint32_t)core_id * ci_chunk;
    uint32_t ci_stop = ci_start + ci_chunk;
    if (ci_stop > Cin) ci_stop = Cin;

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
                int32_t ky_min = max_i32(0,              hx0 - base_h);
                int32_t ky_max = min_i32((int32_t)P - 1, hx1 - base_h);
                if (ky_min > ky_max) continue;

                int32_t kx_min = max_i32(0,              wx0 - base_w);
                int32_t kx_max = min_i32((int32_t)Q - 1, wx1 - base_w);
                if (kx_min > kx_max) continue;

                for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
                    float *dx_ci = pGradIn + (size_t)ci * Hin_t * Win_t;

                    // W[co,ci,:,:] base (assumes layout [Cout][Cin][P][Q])
                    const float *w_co_ci = pWeight
                        + (((size_t)co * (size_t)Cin + (size_t)ci) * (size_t)P * (size_t)Q);

                    for (int32_t ky = ky_min; ky <= ky_max; ++ky) {
                        const int32_t ih = (base_h + ky) - hx0;   // local in tile

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
    uint32_t dim_im_out_x,    // H_out (tile)
    uint32_t dim_im_out_y,    // W_out (tile)
    uint32_t ch_im_out,       // C_out (full)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,        // C_in (full)
    uint32_t dim_kernel_x,    // P
    uint32_t dim_kernel_y,    // Q
    uint32_t stride_h,        // SP
    uint32_t stride_w,        // SQ
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

    // Get core ID for parallelization across input channels
    int8_t core_id = pi_core_id();
    int8_t log2Core = LOG2(NUM_CORES);

    // DEBUG: Print parameters on core 0 only
    if (core_id == 0) {
        printf("\n[ConvGradX] === Kernel Invocation ===\n");
        printf("[ConvGradX] dY tile: H_out=%lu, W_out=%lu, C_out=%lu\n",
               (unsigned long)H_out, (unsigned long)W_out, (unsigned long)C_out);
        printf("[ConvGradX] dX tile: H_in=%lu, W_in=%lu, C_in=%lu\n",
               (unsigned long)H_in, (unsigned long)W_in, (unsigned long)C_in);
        printf("[ConvGradX] Kernel: P=%lu, Q=%lu\n",
               (unsigned long)P, (unsigned long)Q);
        printf("[ConvGradX] Stride: stride_h=%lu, stride_w=%lu\n",
               (unsigned long)stride_h, (unsigned long)stride_w);
        printf("[ConvGradX] Padding: top=%lu, bottom=%lu, left=%lu, right=%lu\n",
               (unsigned long)pad_top, (unsigned long)pad_bottom,
               (unsigned long)pad_left, (unsigned long)pad_right);
        printf("[ConvGradX] Pointers: pGradOut=%p, pWeight=%p, pGradIn=%p\n",
               (void*)pGradOut, (void*)pWeight, (void*)pGradIn);
    }

    // Parallelize over input channels (C_in)
    uint16_t ch_chunk = (C_in >> log2Core) + ((C_in & (NUM_CORES - 1)) != 0);
    uint16_t ch_start = MIN(ch_chunk * core_id, C_in);
    uint16_t ch_stop = MIN(ch_start + ch_chunk, C_in);

    // DEBUG: Print channel assignment for each core
    printf("[ConvGradX] Core %d: ch_range=[%u, %u), chunk_size=%u\n",
           core_id, ch_start, ch_stop, ch_chunk);

    if (ch_stop <= ch_start) {
        printf("[ConvGradX] Core %d: No channels assigned, returning\n", core_id);
        return;
    }

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

    if (core_id == 0) {
        printf("[ConvGradX] Core 0: Initialized dX tile to zero\n");
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
                    int32_t h_in = (int32_t)h_out * (int32_t)stride_h +
                                   (int32_t)kh - (int32_t)pad_top;

                    // Check bounds (tile-local)
                    if (h_in < 0 || h_in >= (int32_t)H_in) {
                        continue;
                    }

                    for (uint32_t w_out = 0; w_out < W_out; ++w_out) {
                        // Compute corresponding input position
                        int32_t w_in = (int32_t)w_out * (int32_t)stride_w +
                                       (int32_t)kw - (int32_t)pad_left;

                        // Check bounds (tile-local)
                        if (w_in < 0 || w_in >= (int32_t)W_in) {
                            continue;
                        }

                        // Accumulate gradient contributions from all output channels
                        // dX index: CHW layout [C_in, H_in, W_in]
                        uint32_t dx_idx = (c_in * H_in + (uint32_t)h_in) * W_in + (uint32_t)w_in;

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

    // DEBUG: Print sample output values on core 0
    if (core_id == 0) {
        printf("[ConvGradX] Core 0: Computation done\n");
        printf("[ConvGradX] Core 0: Sample dX values:\n");
        for (uint32_t i = 0; i < MIN(5, H_in * W_in * C_in); ++i) {
            printf("  dX[%lu] = %.6f\n", (unsigned long)i, pGradIn[i]);
        }

        // Print sample input values
        printf("[ConvGradX] Core 0: Sample dY values:\n");
        for (uint32_t i = 0; i < MIN(5, H_out * W_out * C_out); ++i) {
            printf("  dY[%lu] = %.6f\n", (unsigned long)i, pGradOut[i]);
        }

        printf("[ConvGradX] Core 0: Sample Weight values:\n");
        for (uint32_t i = 0; i < MIN(5, C_out * C_in * P * Q); ++i) {
            printf("  W[%lu] = %.6f\n", (unsigned long)i, pWeight[i]);
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
  // Only supports stride=1
  // No padding support
  // Requires H_out + kernel_size - 1 ≤ H_in

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

  pw_args.skip_wg_grad = 0;  // Compute weight gradient
  pw_args.skip_in_grad = 1;  // Skip input gradient
  pw_args.HWC = 0;           // CHW layout
  pw_args.opt_matmul_type_fw = 0;
  pw_args.opt_matmul_type_wg = 0;
  pw_args.opt_matmul_type_ig = 0;

  pulp_conv_pw_fp32_bw_param_grads_cl(&pw_args);
}


void PULP_PWConvGradX2d_fp32_fp32_fp32_CHW(
    const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
    uint32_t C_out, const float *__restrict__ pWeight, uint32_t C_in,
    float *__restrict__ pGradIn, uint32_t H_in, uint32_t W_in,
    float *__restrict__ pTransposeBuffer, uint32_t transposeBufferSize) {

  struct blob input_blob = {0};
  struct blob output_blob = {0};
  struct blob coeff_blob = {0};

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
  coeff_blob.W = 1;
  coeff_blob.H = 1;
  coeff_blob.C = (int)C_in;
  coeff_blob.dim = (int)(C_out * C_in);

  struct PointWise_Conv_args pw_args;
  memset(&pw_args, 0, sizeof(pw_args));

  pw_args.input = &input_blob;
  pw_args.output = &output_blob;
  pw_args.coeff = &coeff_blob;
  pw_args.transpose_buffer = pTransposeBuffer;

  pw_args.skip_wg_grad = 1;
  pw_args.skip_in_grad = 0;
  pw_args.HWC = 0;
  pw_args.opt_matmul_type_fw = 0;
  pw_args.opt_matmul_type_wg = 0;
  pw_args.opt_matmul_type_ig = 0;

  pulp_conv_pw_fp32_bw_input_grads_cl(&pw_args);
}

// Tile-aware Im2Col-based ConvGradX kernel with offset support
void PULP_ConvGradX2d_fp32_fp32_fp32_CHW_Im2Col_tiled(
    const float *__restrict__ pGradOut,     // dY tile (L1)
    uint32_t dim_im_out_x,                  // dY tile H
    uint32_t dim_im_out_y,                  // dY tile W
    uint32_t ch_im_out,                     // C_out (full)
    const float *__restrict__ pWeight,      // W
    uint32_t ch_im_in,                      // C_in (full)
    uint32_t dim_kernel_x,                  // P (kernel H)
    uint32_t dim_kernel_y,                  // Q (kernel W)
    uint32_t stride_h,                      // stride H
    uint32_t stride_w,                      // stride W
    float *__restrict__ pGradIn,            // dX tile (L1)
    uint32_t dim_im_in_x,                   // dX tile H
    uint32_t dim_im_in_y,                   // dX tile W
    uint32_t padding_y_top,                 // pad top (tile-specific)
    uint32_t padding_y_bottom,              // pad bottom (tile-specific)
    uint32_t padding_x_left,                // pad left (tile-specific)
    uint32_t padding_x_right,               // pad right (tile-specific)
    uint16_t offset_grad_in_h,              // dX tile offset H (global)
    uint16_t offset_grad_in_w,              // dX tile offset W (global)
    uint16_t offset_grad_out_h,             // dY tile offset H (global)
    uint16_t offset_grad_out_w,             // dY tile offset W (global)
    float *__restrict__ ctxtBuffer,
    uint32_t ctxtBufferSize,
    float *__restrict__ btBuffer,
    uint32_t btBufferSize
) {
    const uint32_t Hout_t = dim_im_out_x;
    const uint32_t Wout_t = dim_im_out_y;
    const uint32_t Hin_t  = dim_im_in_x;
    const uint32_t Win_t  = dim_im_in_y;

    const uint32_t Cout = ch_im_out;
    const uint32_t Cin  = ch_im_in;

    const uint32_t P = dim_kernel_x;
    const uint32_t Q = dim_kernel_y;

    const int32_t pad_top  = (int32_t)padding_y_top;
    const int32_t pad_left = (int32_t)padding_x_left;

    const int32_t sh = (int32_t)stride_h;
    const int32_t sw = (int32_t)stride_w;

    const int32_t hx0 = (int32_t)offset_grad_in_h;
    const int32_t wx0 = (int32_t)offset_grad_in_w;
    const int32_t hx1 = hx0 + (int32_t)Hin_t - 1;
    const int32_t wx1 = wx0 + (int32_t)Win_t - 1;

    // Core partition over Cin
    const int core_id = pi_core_id();
    const int ncores  = NUM_CORES;

    const uint32_t ci_chunk = (Cin + (uint32_t)ncores - 1u) / (uint32_t)ncores;
    const uint32_t ci_start = (uint32_t)core_id * ci_chunk;
    uint32_t ci_stop = ci_start + ci_chunk;
    if (ci_stop > Cin) ci_stop = Cin;

    if (ci_start >= ci_stop) {
        return;
    }

    // Initialize output tile to zero
    for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
        float *dx_ci = pGradIn + (size_t)ci * Hin_t * Win_t;
        for (uint32_t ih = 0; ih < Hin_t; ++ih) {
            for (uint32_t iw = 0; iw < Win_t; ++iw) {
                dx_ci[ih * Win_t + iw] = 0.0f;
            }
        }
    }

    // Compute gradient using tile-aware mapping
    for (uint32_t co = 0; co < Cout; ++co) {
        const float *dy_co = pGradOut + (size_t)co * Hout_t * Wout_t;

        for (uint32_t ly = 0; ly < Hout_t; ++ly) {
            const int32_t oy = (int32_t)offset_grad_out_h + (int32_t)ly;
            const int32_t base_h = oy * sh - pad_top;

            for (uint32_t lx = 0; lx < Wout_t; ++lx) {
                const int32_t ox = (int32_t)offset_grad_out_w + (int32_t)lx;
                const int32_t base_w = ox * sw - pad_left;

                const float dy_val = dy_co[ly * Wout_t + lx];

                // Prune kernel positions
                int32_t ky_min = (hx0 > base_h) ? (hx0 - base_h) : 0;
                int32_t ky_max = (hx1 < base_h + (int32_t)P - 1) ? (hx1 - base_h) : ((int32_t)P - 1);
                if (ky_min > ky_max) continue;

                int32_t kx_min = (wx0 > base_w) ? (wx0 - base_w) : 0;
                int32_t kx_max = (wx1 < base_w + (int32_t)Q - 1) ? (wx1 - base_w) : ((int32_t)Q - 1);
                if (kx_min > kx_max) continue;

                for (uint32_t ci = ci_start; ci < ci_stop; ++ci) {
                    float *dx_ci = pGradIn + (size_t)ci * Hin_t * Win_t;

                    // W[co,ci,:,:] base (layout [Cout][Cin][P][Q])
                    const float *w_co_ci = pWeight + (((size_t)co * (size_t)Cin + (size_t)ci) * (size_t)P * (size_t)Q);

                    for (int32_t ky = ky_min; ky <= ky_max; ++ky) {
                        const int32_t ih = (base_h + ky) - hx0;  // local tile coordinate

                        for (int32_t kx = kx_min; kx <= kx_max; ++kx) {
                            const int32_t iw = (base_w + kx) - wx0;  // local tile coordinate

                            dx_ci[(uint32_t)ih * Win_t + (uint32_t)iw] +=
                                dy_val * w_co_ci[(size_t)ky * (size_t)Q + (size_t)kx];
                        }
                    }
                }
            }
        }
    }
}


void PULP_DWConvGradX2d_fp32_fp32_fp32_CHW_tiled(
    const float *__restrict__ pGradOut,
    uint32_t dim_im_out_x,    // H_out (tile)
    uint32_t dim_im_out_y,    // W_out (tile)
    uint32_t ch_im_out,       // C_out (full)  (DW: equals Cin for multiplier=1)
    const float *__restrict__ pWeight,
    uint32_t ch_im_in,        // C_in (full)
    uint32_t dim_kernel_x,    // P (kernel H)
    uint32_t dim_kernel_y,    // Q (kernel W)
    uint32_t stride_h,        // SH
    uint32_t stride_w,        // SW
    float *__restrict__ pGradIn,
    uint32_t dim_im_in_x,     // H_in (tile)
    uint32_t dim_im_in_y,     // W_in (tile)
    uint32_t padding_x_left,  // pad_top
    uint32_t padding_x_right, // pad_bottom (unused here)
    uint32_t padding_y_top,   // pad_left
    uint32_t padding_y_bottom,// pad_right (unused here)
    uint16_t offset_grad_in_h,
    uint16_t offset_grad_in_w,
    uint16_t offset_grad_out_h,
    uint16_t offset_grad_out_w
){
    (void)padding_x_right;
    (void)padding_y_bottom;

    const uint32_t Hout_t = dim_im_out_x;
    const uint32_t Wout_t = dim_im_out_y;
    const uint32_t Hin_t  = dim_im_in_x;
    const uint32_t Win_t  = dim_im_in_y;

    const uint32_t Cout_full = ch_im_out;
    const uint32_t Cin_full  = ch_im_in;

    const uint32_t P = dim_kernel_x;
    const uint32_t Q = dim_kernel_y;

    const int32_t pad_top  = (int32_t)padding_x_left;
    const int32_t pad_left = (int32_t)padding_y_top;

    const int32_t sh = (int32_t)stride_h;
    const int32_t sw = (int32_t)stride_w;

    // dx tile global box [hx0..hx1] x [wx0..wx1]
    const int32_t hx0 = (int32_t)offset_grad_in_h;
    const int32_t wx0 = (int32_t)offset_grad_in_w;
    const int32_t hx1 = hx0 + (int32_t)Hin_t - 1;
    const int32_t wx1 = wx0 + (int32_t)Win_t - 1;

    // -------- core partition over channels --------
    const int core_id = pi_core_id();
    const int ncores  = NUM_CORES;

    const uint32_t c_full = (Cin_full < Cout_full) ? Cin_full : Cout_full; // safety
    const uint32_t c_chunk = (c_full + (uint32_t)ncores - 1u) / (uint32_t)ncores;
    const uint32_t c_start = (uint32_t)core_id * c_chunk;
    uint32_t c_stop = c_start + c_chunk;
    if (c_stop > c_full) c_stop = c_full;

    if (c_start >= c_stop) {
        return;
    }

    for (uint32_t c = c_start; c < c_stop; ++c) {

        float *dx_c       = pGradIn  + (size_t)c * (size_t)Hin_t  * (size_t)Win_t;
        const float *dy_c = pGradOut + (size_t)c * (size_t)Hout_t * (size_t)Wout_t;

        // DW weight layout: [C][1][P][Q] -> contiguous [C][P][Q]
        const float *w_c  = pWeight  + (size_t)c * (size_t)P * (size_t)Q;

        // ---- clear dx tile for this channel ----
        // If your schedule expects accumulation across multiple calls, REMOVE this clear.
        for (uint32_t ih = 0; ih < Hin_t; ++ih) {
            float *row = dx_c + (size_t)ih * (size_t)Win_t;
            for (uint32_t iw = 0; iw < Win_t; ++iw) {
                row[iw] = 0.0f;
            }
        }

        // ---- main scatter from dy tile into dx tile ----
        for (uint32_t ly = 0; ly < Hout_t; ++ly) {
            const int32_t oy = (int32_t)offset_grad_out_h + (int32_t)ly;
            const int32_t base_h = oy * sh - pad_top;

            for (uint32_t lx = 0; lx < Wout_t; ++lx) {
                const int32_t ox = (int32_t)offset_grad_out_w + (int32_t)lx;
                const int32_t base_w = ox * sw - pad_left;

                const float dy_val = dy_c[ly * Wout_t + lx];

                // Intersect kernel footprint with dx tile bounds
                int32_t ky_min = max_i32(0,              hx0 - base_h);
                int32_t ky_max = min_i32((int32_t)P - 1, hx1 - base_h);
                if (ky_min > ky_max) continue;

                int32_t kx_min = max_i32(0,              wx0 - base_w);
                int32_t kx_max = min_i32((int32_t)Q - 1, wx1 - base_w);
                if (kx_min > kx_max) continue;

                for (int32_t ky = ky_min; ky <= ky_max; ++ky) {
                    const int32_t ih = (base_h + ky) - hx0; // local in dx tile

                    for (int32_t kx = kx_min; kx <= kx_max; ++kx) {
                        const int32_t iw = (base_w + kx) - wx0;

                        const size_t w_idx =
                            (size_t)(uint32_t)ky * (size_t)Q +
                            (size_t)(uint32_t)kx;

                        dx_c[(size_t)(uint32_t)ih * (size_t)Win_t + (size_t)(uint32_t)iw] +=
                            dy_val * w_c[w_idx];
                    }
                }
            }
        }
    }
}











