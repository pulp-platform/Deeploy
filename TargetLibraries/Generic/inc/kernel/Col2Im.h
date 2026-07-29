// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __DEEPLOY_BASIC_MATH_COL2IM_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_COL2IM_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                               Col2Im                                       */
/******************************************************************************/

/* Maximum supported number of spatial dimensions. */
#define COL2IM_MAX_SPATIAL_DIMS 4

/*
 * DECLARE_COL2IM_FN(SUFFIX, DATA_TYPE)
 *
 * Emits a forward declaration for Col2Im_<SUFFIX>.
 * The matching definition lives in Col2Im.c via DEFINE_COL2IM_FN.
 *
 * Implements ONNX Col2Im semantics:
 *   input  : (N, C * prod(block_shape), L)  — column matrix
 *   output : (N, C, image_shape[0], ..., image_shape[P-1])
 *
 * For each kernel position bk and output block ob the contribution is
 * accumulated into the corresponding image location (with bounds checking
 * to handle padding).  The output is zero-initialised before accumulation.
 *
 * pads layout: [p_0_begin, ..., p_{P-1}_begin, p_0_end, ..., p_{P-1}_end]
 */
#define DECLARE_COL2IM_FN(SUFFIX, DATA_TYPE)                                   \
  void Col2Im_##SUFFIX(const DATA_TYPE *input, DATA_TYPE *output, int32_t N,   \
                       int32_t C, int32_t spatial_dims,                        \
                       const int32_t *image_shape, const int32_t *block_shape, \
                       const int32_t *dilations, const int32_t *pads,          \
                       const int32_t *strides)

DECLARE_COL2IM_FN(fp32, float32_t);
DECLARE_COL2IM_FN(s8, int8_t);
DECLARE_COL2IM_FN(u8, uint8_t);

#endif //__DEEPLOY_BASIC_MATH_COL2IM_KERNEL_HEADER_
