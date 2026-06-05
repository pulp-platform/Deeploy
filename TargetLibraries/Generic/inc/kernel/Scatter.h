/*
 * SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __DEEPLOY_BASIC_MATH_SCATTER_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_SCATTER_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/******************************************************************************/
/*                              Scatter                                       */
/******************************************************************************/

/* Maximum supported tensor rank. */
#define SCATTER_MAX_NDIM 8

/* Reduction modes (mirrors ONNX ScatterElements `reduction` attribute). */
#define SCATTER_REDUCTION_NONE 0
#define SCATTER_REDUCTION_ADD 1
#define SCATTER_REDUCTION_MUL 2
#define SCATTER_REDUCTION_MIN 3
#define SCATTER_REDUCTION_MAX 4

/*
 * DECLARE_SCATTER_FN(SUFFIX, DATA_TYPE)
 *
 * Emits a forward declaration for Scatter_<SUFFIX>.
 * The matching definition lives in Scatter.c via DEFINE_SCATTER_FN.
 */
#define DECLARE_SCATTER_FN(SUFFIX, DATA_TYPE)                                  \
  void Scatter_##SUFFIX(                                                       \
      const DATA_TYPE *data, const int32_t *indices, const DATA_TYPE *updates, \
      DATA_TYPE *output, int32_t ndim, const int32_t *data_shape,              \
      const int32_t *indices_shape, int32_t axis, int32_t reduction)

DECLARE_SCATTER_FN(fp32, float32_t);
DECLARE_SCATTER_FN(s8, int8_t);
DECLARE_SCATTER_FN(u8, uint8_t);

#endif //__DEEPLOY_BASIC_MATH_SCATTER_KERNEL_HEADER_
