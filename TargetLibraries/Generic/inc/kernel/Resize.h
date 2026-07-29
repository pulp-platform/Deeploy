// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#ifndef __DEEPLOY_BASIC_MATH_RESIZE_KERNEL_HEADER_
#define __DEEPLOY_BASIC_MATH_RESIZE_KERNEL_HEADER_

#include "DeeployBasicMath.h"

/* Maximum number of spatial dimensions (excludes batch N and channels C). */
#define RESIZE_MAX_SPATIAL_DIMS 4

/* Element type — passed as a compile-time constant from generated code. */
typedef enum {
  RESIZE_TYPE_FLOAT32 = 0,
  RESIZE_TYPE_INT8,
  RESIZE_TYPE_UINT8,
  RESIZE_TYPE_INT16,
  RESIZE_TYPE_UINT16,
  RESIZE_TYPE_INT32,
  RESIZE_TYPE_UINT32,
} resize_type_t;

/* Interpolation mode (mirrors ONNX Resize `mode` attribute). */
typedef enum {
  RESIZE_MODE_NEAREST = 0,
  RESIZE_MODE_LINEAR,
  RESIZE_MODE_CUBIC,
} resize_mode_t;

/* Coordinate transformation mode. */
typedef enum {
  RESIZE_COORD_ASYMMETRIC = 0,
  RESIZE_COORD_HALF_PIXEL,
  RESIZE_COORD_HALF_PIXEL_SYMMETRIC,
  RESIZE_COORD_PYTORCH_HALF_PIXEL,
  RESIZE_COORD_ALIGN_CORNERS,
  RESIZE_COORD_TF_CROP_AND_RESIZE,
} resize_coord_mode_t;

/* Nearest-neighbour rounding mode. */
typedef enum {
  RESIZE_NEAREST_FLOOR = 0,
  RESIZE_NEAREST_CEIL,
  RESIZE_NEAREST_ROUND_PREFER_FLOOR,
  RESIZE_NEAREST_ROUND_PREFER_CEIL,
} resize_nearest_mode_t;

/*
 * Resize — single function for all element types.
 *
 *   input / output – NCHW tensors (void* to stay type-agnostic)
 *   type_tag       – element type; drives element size and float conversion
 *   N, C           – batch size and number of channels
 *   spatial_dims   – number of spatial dimensions (1..RESIZE_MAX_SPATIAL_DIMS)
 *   input_shape    – spatial sizes of the input  [d0, d1, …]
 *   output_shape   – spatial sizes of the output [d0, d1, …]
 *   mode           – interpolation mode
 *   coord_mode     – coordinate transformation mode
 *   nearest_mode   – rounding mode (only used when mode == RESIZE_MODE_NEAREST)
 */
void Resize(const void *input, void *output, resize_type_t type_tag, int32_t N,
            int32_t C, int32_t spatial_dims, const int32_t *input_shape,
            const int32_t *output_shape, resize_mode_t mode,
            resize_coord_mode_t coord_mode, resize_nearest_mode_t nearest_mode);

#endif // __DEEPLOY_BASIC_MATH_RESIZE_KERNEL_HEADER_
