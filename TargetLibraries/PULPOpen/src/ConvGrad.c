
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

// Define only the structures we need from pulp-trainlib
struct blob {
    float *data;
    float *diff;
    int dim;
    int W;
    int H;
    int C;
};

struct Conv2D_args {
    struct blob * input;
    struct blob * coeff;
    struct blob * bias;
    struct blob * output;
    int Lpad;
    int Rpad;
    int Upad;
    int Dpad;
    int stride_h;
    int stride_w;
    float * i2c_buffer;
    float * bt_buffer;
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

// Forward declare the pulp-trainlib function we need
void pulp_conv2d_fp32_bw_param_grads_cl(void * Conv2D_args);
void pulp_conv2d_fp32_bw_input_grads_cl(void * Conv2D_args);

void PULP_ConvGradW2d_fp32_fp32_fp32_NCHW_trainlib(
    const float *__restrict__ pGradOut,
    uint32_t H_out, uint32_t W_out, uint32_t C_out,
    const float *__restrict__ pInput,
    uint32_t H_in, uint32_t W_in, uint32_t C_in,
    uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradWeight,
    uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right)
{
  memset(pGradWeight, 0, sizeof(float) * (C_out * C_in * P * Q));

  struct blob input_blob  = {0};
  struct blob output_blob = {0};
  struct blob coeff_blob  = {0};
  struct blob bias_blob   = {0};

  input_blob.data = (float*)pInput;
  input_blob.diff = NULL;                
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_in;
  input_blob.dim = (int)(C_in * H_in * W_in);

  output_blob.data = NULL;
  output_blob.diff = (float*)pGradOut;
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_out;
  output_blob.dim = (int)(C_out * H_out * W_out);

  coeff_blob.data = NULL;
  coeff_blob.diff = (float*)pGradWeight;
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

  conv_args.input  = &input_blob;
  conv_args.output = &output_blob;
  conv_args.coeff  = &coeff_blob;
  conv_args.bias   = &bias_blob;

  conv_args.Lpad = (int)pad_left;
  conv_args.Rpad = (int)pad_right;
  conv_args.Upad = (int)pad_top;
  conv_args.Dpad = (int)pad_bottom;
  conv_args.stride_h = (int)SP;
  conv_args.stride_w = (int)SQ;

  conv_args.i2c_buffer = NULL;            
  conv_args.bt_buffer  = NULL;

  conv_args.skip_wg_grad = 0;            
  conv_args.skip_in_grad = 1;             
  conv_args.HWC = 0;                      
  conv_args.USE_BIASES = 0;
  conv_args.USE_IM2COL = 0;               
  conv_args.USE_DMA_IM2COL = 0;

  pulp_conv2d_fp32_bw_param_grads_cl(&conv_args);

}


void PULP_ConvGradX2d_fp32_fp32_fp32_NCHW_trainlib(
    const float *__restrict__ pGradOut,
    uint32_t H_out, uint32_t W_out, uint32_t C_out,
    const float *__restrict__ pWeight,
    uint32_t C_in,
    uint32_t P, uint32_t Q,
    uint32_t SP, uint32_t SQ,
    float *__restrict__ pGradIn,
    uint32_t H_in, uint32_t W_in,
    uint32_t pad_top, uint32_t pad_bottom,
    uint32_t pad_left, uint32_t pad_right)
{
  // dX 清零（保险）
  memset(pGradIn, 0, sizeof(float) * (C_in * H_in * W_in));

  struct blob input_blob  = (struct blob){0};
  struct blob output_blob = (struct blob){0};
  struct blob coeff_blob  = (struct blob){0};
  struct blob bias_blob   = (struct blob){0};

  // ✅ trainlib bw_input_grads naive: A=input->diff 写 dX
  input_blob.data = NULL;
  input_blob.diff = (float*)pGradIn;        // ✅ dX 放这里
  input_blob.W = (int)W_in;
  input_blob.H = (int)H_in;
  input_blob.C = (int)C_in;
  input_blob.dim = (int)(C_in * H_in * W_in);

  // ✅ trainlib bw_input_grads naive: C=output->diff 读 dY
  output_blob.data = NULL;
  output_blob.diff = (float*)pGradOut;      // ✅ dY 放这里
  output_blob.W = (int)W_out;
  output_blob.H = (int)H_out;
  output_blob.C = (int)C_out;
  output_blob.dim = (int)(C_out * H_out * W_out);

  // weights（只用 data 指针 + W/H）
  coeff_blob.data = (float*)pWeight;
  coeff_blob.diff = NULL;
  coeff_blob.W = (int)Q;
  coeff_blob.H = (int)P;
  coeff_blob.C = (int)C_out;                // 这里对 IG naive 分支基本不关键
  coeff_blob.dim = (int)(C_out * C_in * P * Q);

  bias_blob.data = NULL;
  bias_blob.diff = NULL;
  bias_blob.W = 1; bias_blob.H = 1;
  bias_blob.C = (int)C_out;
  bias_blob.dim = (int)C_out;

  struct Conv2D_args conv_args;
  memset(&conv_args, 0, sizeof(conv_args));

  conv_args.input  = &input_blob;
  conv_args.output = &output_blob;
  conv_args.coeff  = &coeff_blob;
  conv_args.bias   = &bias_blob;

  conv_args.Lpad = (int)pad_left;
  conv_args.Rpad = (int)pad_right;
  conv_args.Upad = (int)pad_top;
  conv_args.Dpad = (int)pad_bottom;
  conv_args.stride_h = (int)SP;
  conv_args.stride_w = (int)SQ;

  conv_args.i2c_buffer = NULL;
  conv_args.bt_buffer  = NULL;

  conv_args.skip_wg_grad = 1;
  conv_args.skip_in_grad = 0;
  conv_args.HWC = 0;
  conv_args.USE_BIASES = 0;
  conv_args.USE_IM2COL = 0;       
  conv_args.USE_DMA_IM2COL = 0;

  pulp_conv2d_fp32_bw_input_grads_cl(&conv_args);
  printf("gradin is %f\n", pGradIn[0]);
}


// ============================================================================
// void PULP_ConvTrans2d_fp32_fp32_fp32_HWC




// void PULP_ConvTrans2d_fp32_fp32_fp32_HWC(
//     const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
//     uint32_t F_total, const float *__restrict__ pWeight, uint32_t C, uint32_t P,
//     uint32_t Q, uint32_t SP, uint32_t SQ, float *__restrict__ pGradIn,
//     uint32_t pad_top, uint32_t pad_bottom, uint32_t pad_left,
//     uint32_t pad_right) {
//   int8_t core_id = pi_core_id();
//   int8_t log2Core = LOG2(NUM_CORES);

//   uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
//   uint16_t ch_start = MIN(ch_chunk * core_id, C);
//   uint16_t ch_stop = MIN(ch_start + ch_chunk, C);
//   uint16_t ch_count = ch_stop - ch_start;

//   if (ch_count == 0) {
//     return;
//   }

//   uint32_t H_in = (H_out - 1) * SP + P - pad_top - pad_bottom;
//   uint32_t W_in = (W_out - 1) * SQ + Q - pad_left - pad_right;

//   for (uint32_t ih = 0; ih < H_in; ++ih) {
//     for (uint32_t iw = 0; iw < W_in; ++iw) {
//       uint32_t gi_base = (ih * W_in + iw) * C;
//       for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {
//         pGradIn[gi_base + ic] = 0.0f;
//       }
//     }
//   }

//   for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {

//     uint32_t oc = ic;

//     for (uint32_t kh = 0; kh < P; ++kh) {
//       for (uint32_t kw = 0; kw < Q; ++kw) {

//         uint32_t w_idx = ic * (P * Q) + kh * Q + kw;

//         for (uint32_t oh = 0; oh < H_out; ++oh) {
//           int32_t ih =
//               (int32_t)oh * (int32_t)SP + (int32_t)kh - (int32_t)pad_top;

//           if (ih < 0 || ih >= (int32_t)H_in)
//             continue;

//           for (uint32_t ow = 0; ow < W_out; ++ow) {
//             int32_t iw =
//                 (int32_t)ow * (int32_t)SQ + (int32_t)kw - (int32_t)pad_left;

//             if (iw < 0 || iw >= (int32_t)W_in)
//               continue;

//             uint32_t go_idx = (oh * W_out + ow) * C + oc;

//             uint32_t gi_idx = ((uint32_t)ih * W_in + (uint32_t)iw) * C + ic;

//             pGradIn[gi_idx] += pGradOut[go_idx] * pWeight[w_idx];
//           }
//         }
//       }
//     }
//   }
// }

// void PULP_DWConvTrans2d_fp32_fp32_fp32_HWC(
//     const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
//     uint32_t C_total, const float *__restrict__ pWeight,
//     uint32_t P, uint32_t Q, uint32_t SP, uint32_t SQ,
//     float *__restrict__ pGradIn, uint32_t pad_top, uint32_t pad_bottom,
//     uint32_t pad_left, uint32_t pad_right) {

//   uint32_t C = C_total;

//   int8_t core_id = pi_core_id();
//   int8_t log2Core = LOG2(NUM_CORES);

//   uint16_t ch_chunk = (C >> log2Core) + ((C & (NUM_CORES - 1)) != 0);
//   uint16_t ch_start = MIN(ch_chunk * core_id, C);
//   uint16_t ch_stop = MIN(ch_start + ch_chunk, C);
//   uint16_t ch_count = ch_stop - ch_start;

//   if (ch_count == 0) {
//     return;
//   }

//   uint32_t H_in = (H_out - 1) * SP + P - pad_top - pad_bottom;
//   uint32_t W_in = (W_out - 1) * SQ + Q - pad_left - pad_right;

//   for (uint32_t ih = 0; ih < H_in; ++ih) {
//     for (uint32_t iw = 0; iw < W_in; ++iw) {
//       uint32_t gi_base = (ih * W_in + iw) * C;
//       for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {
//         pGradIn[gi_base + ic] = 0.0f;
//       }
//     }
//   }
//   for (uint32_t ic = ch_start; ic < ch_stop; ++ic) {

//     uint32_t oc = ic; 

//     for (uint32_t kh = 0; kh < P; ++kh) {
//       for (uint32_t kw = 0; kw < Q; ++kw) {

//         uint32_t w_idx = ic * (P * Q) + kh * Q + kw;
//         float w_val = pWeight[w_idx];

//         for (uint32_t oh = 0; oh < H_out; ++oh) {
//           int32_t ih =  (int32_t)oh * (int32_t)SP + (int32_t)kh - (int32_t)pad_top;
//           if (ih < 0 || ih >= (int32_t)H_in) continue;

//           for (uint32_t ow = 0; ow < W_out; ++ow) {
//             int32_t iw =  (int32_t)ow * (int32_t)SQ + (int32_t)kw - (int32_t)pad_left;
//             if (iw < 0 || iw >= (int32_t)W_in) continue;

//             uint32_t go_idx = (oh * W_out + ow) * C + oc;
//             uint32_t gi_idx = ((uint32_t)ih * W_in + (uint32_t)iw) * C + ic;
            
//             // Workaround for GCC/RISC-V compiler optimization bug
//             // Without this printf, the compiler generates incorrect pointer arithmetic
//             // causing wrong results at specific indices (w=0,1 positions)
//             printf("hello");
//             pGradIn[gi_idx] += pGradOut[go_idx] * w_val;
//           }
//         }
//       }
//     }
//   }
// }

// void PULP_ConvGradW2d_fp32_fp32_fp32_NCHW(
//     const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
//     uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
//     uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
//     uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
//     uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
//   int8_t core_id = pi_core_id();
//   int8_t log2Core = LOG2(NUM_CORES);

//   uint16_t ch_chunk = (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
//   uint16_t ch_start = MIN(ch_chunk * core_id, C_out);
//   uint16_t ch_stop = MIN(ch_start + ch_chunk, C_out);

//   if (ch_start >= ch_stop) {
//     return;
//   }

//   // Compute weight gradients
//   for (uint32_t oc = ch_start; oc < ch_stop; ++oc) {
//     for (uint32_t ic = 0; ic < C_in; ++ic) {
//       for (uint32_t kh = 0; kh < P; ++kh) {
//         for (uint32_t kw = 0; kw < Q; ++kw) {

//           float grad_sum = 0.0f;
//           int valid_count = 0;

//           for (uint32_t oh = 0; oh < H_out; ++oh) {
//             for (uint32_t ow = 0; ow < W_out; ++ow) {

//               int32_t ih =
//                   (int32_t)oh * (int32_t)SP + (int32_t)kh - (int32_t)pad_top;
//               int32_t iw =
//                   (int32_t)ow * (int32_t)SQ + (int32_t)kw - (int32_t)pad_left;

//               if (ih >= 0 && ih < (int32_t)H_in && iw >= 0 &&
//                   iw < (int32_t)W_in) {
//                 uint32_t go_idx = (oc * H_out + oh) * W_out + ow;
//                 float gy = pGradOut[go_idx];

//                 uint32_t in_idx = (ic * H_in + ih) * W_in + iw;
//                 float x = pInput[in_idx];

//                 grad_sum += gy * x;
//                 valid_count++;
//               }
//             }
//           }

//           uint32_t gw_idx = ((oc * C_in + ic) * P + kh) * Q + kw;
//           pGradWeight[gw_idx] = grad_sum;
//         }
//       }
//     }
//   }
// }

// void PULP_ConvGradB2d_fp32_fp32_NCHW(const float *__restrict__ pGradOut,
//                                      uint32_t H_out, uint32_t W_out,
//                                      uint32_t C_out,
//                                      float *__restrict__ pGradBias) {
//   int8_t core_id = pi_core_id();
//   int8_t log2Core = LOG2(NUM_CORES);

//   uint16_t ch_chunk = (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
//   uint16_t ch_start = MIN(ch_chunk * core_id, C_out);
//   uint16_t ch_stop = MIN(ch_start + ch_chunk, C_out);

//   if (ch_start >= ch_stop) {
//     return;
//   }

//   for (uint32_t oc = ch_start; oc < ch_stop; ++oc) {
//     float grad_sum = 0.0f;

//     // Sum over all spatial positions
//     for (uint32_t oh = 0; oh < H_out; ++oh) {
//       for (uint32_t ow = 0; ow < W_out; ++ow) {
//         // NCHW layout: [oc, oh, ow]
//         uint32_t go_idx = (oc * H_out + oh) * W_out + ow;
//         grad_sum += pGradOut[go_idx];
//       }
//     }

//     pGradBias[oc] = grad_sum;
//   }
// }


// void PULP_DWConvGradW2d_fp32_fp32_fp32_NCHW(
//     const float *__restrict__ pGradOut, uint32_t H_out, uint32_t W_out,
//     uint32_t C_out, const float *__restrict__ pInput, uint32_t H_in,
//     uint32_t W_in, uint32_t C_in, uint32_t P, uint32_t Q, uint32_t SP,
//     uint32_t SQ, float *__restrict__ pGradWeight, uint32_t pad_top,
//     uint32_t pad_bottom, uint32_t pad_left, uint32_t pad_right) {
  
//   int8_t core_id = pi_core_id();
//   int8_t log2Core = LOG2(NUM_CORES);

//   uint16_t ch_chunk = (C_out >> log2Core) + ((C_out & (NUM_CORES - 1)) != 0);
//   uint16_t ch_start = MIN(ch_chunk * core_id, C_out);
//   uint16_t ch_stop  = MIN(ch_start + ch_chunk, C_out);

//   if (ch_start >= ch_stop) {
//     return;
//   }

//   uint32_t C_in_per_group = C_in / C_out;

//   for (uint32_t oc = ch_start; oc < ch_stop; ++oc) {
//     uint32_t ic_start = oc * C_in_per_group;
    
//     for (uint32_t ic_idx = 0; ic_idx < C_in_per_group; ++ic_idx) {
//       uint32_t ic = ic_start + ic_idx;
      
//       for (uint32_t kh = 0; kh < P; ++kh) {
//         for (uint32_t kw = 0; kw < Q; ++kw) {

//           float grad_sum = 0.0f;

//           for (uint32_t oh = 0; oh < H_out; ++oh) {
//             for (uint32_t ow = 0; ow < W_out; ++ow) {

//               int32_t ih = (int32_t)oh * (int32_t)SP +
//                            (int32_t)kh - (int32_t)pad_top;
//               int32_t iw = (int32_t)ow * (int32_t)SQ +
//                            (int32_t)kw - (int32_t)pad_left;

//               if (ih >= 0 && ih < (int32_t)H_in &&
//                   iw >= 0 && iw < (int32_t)W_in) {

//                 uint32_t go_idx = (oc * H_out + oh) * W_out + ow;
//                 float gy = pGradOut[go_idx];

//                 uint32_t in_idx =
//                     (ic * H_in + (uint32_t)ih) * W_in + (uint32_t)iw;

//                 grad_sum += gy * pInput[in_idx];
//                 printf("hello");
//               }
//             }
//           }

//           uint32_t gw_idx =
//               ((oc * C_in_per_group + ic_idx) * P + kh) * Q + kw;
//           pGradWeight[gw_idx] = grad_sum;


//         }
//       }
//     }
//   }
// }

