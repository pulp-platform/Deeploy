/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "kernel/PULPDWConv3x3.h"

#include "pmsis.h"
#include "pulp_nn_kernels.h"
#include "pulp_nn_utils.h"

#ifndef NUM_CORES
#define NUM_CORES 8
#endif

#define DW3X3_MIN(a, b) (((a) < (b)) ? (a) : (b))

/*
 * Shape-specialised 3x3 / stride 1 / pad 1 depthwise convolution.
 *
 * PULP-NN's generic NxM depthwise kernel spends ~37 instructions per output
 * pixel for 9 MACs: it rebuilds an im2col strip per output column, splits the
 * 9 taps into "two 4-way dot products plus a scalar leftover", re-derives the
 * quantisation operands per pixel, and re-tests the padding conditions inside
 * the pixel loop.
 *
 * This variant follows the structure the GAP9 SDK's Autotiler uses in
 * KerConvDW3x3Stride1_Body_SQ8 (tools/autotiler_v3/CNN_Libraries_SQ8,
 * Apache-2.0):
 *
 *   1. the 9 taps are held in three v4s registers with lane 3 zeroed, loaded
 *      once per channel, so the tap loop becomes three clean 4-way dot
 *      products with no scalar leftover path;
 *   2. the three input rows are rotated through registers (V0<-V1<-V2), so
 *      each output pixel costs a single 4-byte load instead of rebuilding an
 *      im2col strip;
 *   3. the per-channel quantisation operands are hoisted out of the pixel
 *      loop and the requantisation is inlined;
 *   4. the column borders are folded into shifted weight vectors and the row
 *      borders into a zero register, instead of zero-filling a scratch buffer.
 *
 * Measured on MLPerf Tiny MobileNetV1 (VisualWakeWords, 96x96) on GAP9/GVSoC
 * with 8 cores: the eight layers that take this path go from 649712 to 193646
 * cycles (3.36x), bit-exact. Everything else falls through to the generic
 * PULP-NN kernel unchanged.
 */

static inline uint8_t __attribute__((always_inline))
DeeployPULP_dw_requant_u8(int32_t acc, int32_t kappa, int32_t lambda,
                          uint16_t shift) {
  return (uint8_t)clip8(((kappa * acc) + lambda) >> shift);
}

void DeeployPULP_DW_Conv2d_3x3_u8_u8_i8(
    uint8_t *pIn, uint8_t *pIm2ColBuffer, int8_t *pBias, uint8_t *pOut,
    int8_t *pWeight, int8_t *pWtBuffer, int32_t *pKappa, int32_t *pLambda,
    uint16_t out_mult, uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y,
    uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
    uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_y_top,
    uint16_t padding_y_bottom, uint16_t padding_x_left,
    uint16_t padding_x_right, uint16_t stride_x, uint16_t stride_y,
    uint8_t flag_relu, uint8_t flag_batch_norm) {

  if (!(dim_kernel_x == 3 && dim_kernel_y == 3 && stride_x == 1 &&
        stride_y == 1 && padding_x_left == 1 && padding_x_right == 1 &&
        padding_y_top == 1 && padding_y_bottom == 1 && dim_out_x == dim_in_x &&
        dim_out_y == dim_in_y && dim_in_x >= 4 && pBias == NULL && flag_relu &&
        flag_batch_norm)) {
    pulp_nn_depthwise_u8_u8_i8(
        pIn, pIm2ColBuffer, pBias, pOut, pWeight, pWtBuffer, pKappa, pLambda,
        out_mult, out_shift, dim_in_x, dim_in_y, ch_in, dim_out_x, dim_out_y,
        ch_out, dim_kernel_x, dim_kernel_y, padding_y_top, padding_y_bottom,
        padding_x_left, padding_x_right, stride_x, stride_y, flag_relu,
        flag_batch_norm);
    return;
  }

  const uint8_t core_id = pi_core_id();
  const int chunk =
      (ch_out >> __builtin_ctz(NUM_CORES)) + ((ch_out & (NUM_CORES - 1)) != 0);
  const int start_channel = DW3X3_MIN(chunk * core_id, ch_out);
  const int stop_channel = DW3X3_MIN(start_channel + chunk, ch_out);

  const v4u ZERO = (v4u){0, 0, 0, 0};
  const int W = dim_in_x, H = dim_in_y, OS = dim_out_x * ch_out;
  const int plane = dim_in_x * dim_in_y;

  for (int c = start_channel; c < stop_channel; c++) {
    const uint8_t *inp = pIn + c * plane;
    const int8_t *wt = pWeight + c * 9;
    const int32_t kk = pKappa[c], ll = pLambda[c];

    /* interior column x: taps at x-1,x,x+1 -> read 4B at x-1, lane 3 zeroed */
    const v4s C0 = (v4s){wt[0], wt[1], wt[2], 0};
    const v4s C1 = (v4s){wt[3], wt[4], wt[5], 0};
    const v4s C2 = (v4s){wt[6], wt[7], wt[8], 0};
    /* x == 0: read 4B at 0, tap 0 sits on the pad -> drop it */
    const v4s L0 = (v4s){wt[1], wt[2], 0, 0};
    const v4s L1 = (v4s){wt[4], wt[5], 0, 0};
    const v4s L2 = (v4s){wt[7], wt[8], 0, 0};
    /* x == W-2: read 4B at W-4 so the load never runs past the row */
    const v4s M0 = (v4s){0, wt[0], wt[1], wt[2]};
    const v4s M1 = (v4s){0, wt[3], wt[4], wt[5]};
    const v4s M2 = (v4s){0, wt[6], wt[7], wt[8]};
    /* x == W-1: read 4B at W-4, tap 2 sits on the pad -> drop it */
    const v4s R0 = (v4s){0, 0, wt[0], wt[1]};
    const v4s R1 = (v4s){0, 0, wt[3], wt[4]};
    const v4s R2 = (v4s){0, 0, wt[6], wt[7]};

    for (int x = 0; x < W; x++) {
      v4s w0, w1, w2;
      int off;
      if (x == 0) {
        w0 = L0;
        w1 = L1;
        w2 = L2;
        off = 0;
      } else if (x < W - 2) {
        w0 = C0;
        w1 = C1;
        w2 = C2;
        off = x - 1;
      } else if (x == W - 2) {
        w0 = M0;
        w1 = M1;
        w2 = M2;
        off = W - 4;
      } else {
        w0 = R0;
        w1 = R1;
        w2 = R2;
        off = W - 4;
      }

      const uint8_t *pi = inp + off;
      uint8_t *po = pOut + c + x * ch_out;
      v4u V0 = ZERO; /* input row -1 is the top pad          */
      v4u V1 = *(v4u *)pi;
      pi += W; /* input row 0                          */

      for (int y = 0; y < H - 1; y++) {
        v4u V2 = *(v4u *)pi;
        pi += W; /* the only load per output pixel     */
        int acc = SumDotp4(V0, w0, 0);
        acc = SumDotp4(V1, w1, acc);
        acc = SumDotp4(V2, w2, acc);
        *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
        po += OS;
        V0 = V1;
        V1 = V2; /* two of the three rows are reused next pixel    */
      }
      { /* last output row: input row H is the bottom pad, w2 contributes 0 */
        int acc = SumDotp4(V0, w0, 0);
        acc = SumDotp4(V1, w1, acc);
        *po = DeeployPULP_dw_requant_u8(acc, kk, ll, out_shift);
      }
    }
  }
  pi_cl_team_barrier();
}
