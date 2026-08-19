/*
 * SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_MATH_PULP_DWCONV3X3_KERNEL_HEADER_
#define __DEEPLOY_MATH_PULP_DWCONV3X3_KERNEL_HEADER_

#include <stdint.h>

/*
 * Shape-specialised 3x3 / stride 1 / pad 1 depthwise convolution. Same
 * signature as pulp_nn_depthwise_u8_u8_i8, to which it falls back for every
 * other shape, so the code generator only has to change the emitted name.
 * See TargetLibraries/PULPOpen/src/PULPDWConv3x3.c for the rationale.
 */
void DeeployPULP_DW_Conv2d_3x3_u8_u8_i8(
    uint8_t *pIn, uint8_t *pIm2ColBuffer, int8_t *pBias, uint8_t *pOut,
    int8_t *pWeight, int8_t *pWtBuffer, int32_t *pKappa, int32_t *pLambda,
    uint16_t out_mult, uint16_t out_shift, uint16_t dim_in_x, uint16_t dim_in_y,
    uint16_t ch_in, uint16_t dim_out_x, uint16_t dim_out_y, uint16_t ch_out,
    uint16_t dim_kernel_x, uint16_t dim_kernel_y, uint16_t padding_y_top,
    uint16_t padding_y_bottom, uint16_t padding_x_left,
    uint16_t padding_x_right, uint16_t stride_x, uint16_t stride_y,
    uint8_t flag_relu, uint8_t flag_batch_norm);

#endif // __DEEPLOY_MATH_PULP_DWCONV3X3_KERNEL_HEADER_
