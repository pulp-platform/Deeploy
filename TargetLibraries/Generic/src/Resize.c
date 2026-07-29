// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

#include <math.h>

#include "DeeployBasicMath.h"

/* Number of bytes per element. */
static inline uint32_t _resize_element_size(resize_type_t type_tag) {
  switch (type_tag) {
  case RESIZE_TYPE_FLOAT32:
    return sizeof(float32_t);
  case RESIZE_TYPE_INT16:
  case RESIZE_TYPE_UINT16:
    return sizeof(int16_t);
  case RESIZE_TYPE_INT32:
  case RESIZE_TYPE_UINT32:
    return sizeof(int32_t);
  default: /* INT8, UINT8 */
    return sizeof(int8_t);
  }
}

/* Read one element as float for use in the linear interpolation path. */
static inline float32_t _resize_read(const void *buf, int32_t idx,
                                     resize_type_t type_tag) {
  switch (type_tag) {
  case RESIZE_TYPE_FLOAT32:
    return ((const float32_t *)buf)[idx];
  case RESIZE_TYPE_INT8:
    return (float32_t)((const int8_t *)buf)[idx];
  case RESIZE_TYPE_UINT8:
    return (float32_t)((const uint8_t *)buf)[idx];
  case RESIZE_TYPE_INT16:
    return (float32_t)((const int16_t *)buf)[idx];
  case RESIZE_TYPE_UINT16:
    return (float32_t)((const uint16_t *)buf)[idx];
  case RESIZE_TYPE_INT32:
    return (float32_t)((const int32_t *)buf)[idx];
  default: /* RESIZE_TYPE_UINT32 */
    return (float32_t)((const uint32_t *)buf)[idx];
  }
}

/* Round val to the nearest integer, breaking ties toward the nearest even
 * integer (banker's rounding). This matches numpy.round / the ONNX reference
 * implementation. */
static inline float32_t _round_half_to_even(float32_t val) {
  float32_t f = floorf(val);
  float32_t diff = val - f;
  if (diff < 0.5f)
    return f;
  if (diff > 0.5f)
    return f + 1.0f;
  /* exactly 0.5: pick the even neighbour */
  return (fmodf(f, 2.0f) == 0.0f) ? f : f + 1.0f;
}

/* Write a float result back as the element's native type. */
static inline void _resize_write(void *buf, int32_t idx, float32_t val,
                                 resize_type_t type_tag) {
  switch (type_tag) {
  case RESIZE_TYPE_FLOAT32:
    ((float32_t *)buf)[idx] = val;
    break;
  case RESIZE_TYPE_INT8:
    ((int8_t *)buf)[idx] = (int8_t)_round_half_to_even(val);
    break;
  case RESIZE_TYPE_UINT8:
    ((uint8_t *)buf)[idx] = (uint8_t)_round_half_to_even(val);
    break;
  case RESIZE_TYPE_INT16:
    ((int16_t *)buf)[idx] = (int16_t)_round_half_to_even(val);
    break;
  case RESIZE_TYPE_UINT16:
    ((uint16_t *)buf)[idx] = (uint16_t)_round_half_to_even(val);
    break;
  case RESIZE_TYPE_INT32:
    ((int32_t *)buf)[idx] = (int32_t)_round_half_to_even(val);
    break;
  default: /* RESIZE_TYPE_UINT32 */
    ((uint32_t *)buf)[idx] = (uint32_t)_round_half_to_even(val);
    break;
  }
}

/* Map an output coordinate to its source coordinate in the input. */
static float32_t _resize_get_coord(int32_t out_idx, int32_t in_size,
                                   int32_t out_size,
                                   resize_coord_mode_t coord_mode) {
  float32_t x_scale = (float32_t)out_size / (float32_t)in_size;
  switch (coord_mode) {

  case RESIZE_COORD_HALF_PIXEL:
    return ((float32_t)out_idx + 0.5f) / x_scale - 0.5f;

  case RESIZE_COORD_HALF_PIXEL_SYMMETRIC: {
    float32_t adjustment =
        (float32_t)out_size / (floorf((float32_t)in_size + 0.5f) * x_scale);
    return ((float32_t)out_idx + 0.5f) / x_scale * adjustment - 0.5f;
  }
  case RESIZE_COORD_ALIGN_CORNERS:
    if (out_size == 1)
      return 0.0f;
    return (float32_t)out_idx * (float32_t)(in_size - 1) /
           (float32_t)(out_size - 1);

  case RESIZE_COORD_PYTORCH_HALF_PIXEL:
    if (out_size == 1)
      return 0.0f;
    return ((float32_t)out_idx + 0.5f) / x_scale - 0.5f;

  default: /* RESIZE_COORD_ASYMMETRIC */
    return (float32_t)out_idx / x_scale;
  }
}

/* Round a source coordinate to the nearest input index and clamp to [0,
 * max_idx]. */
static int32_t _resize_nearest_idx(float32_t x, int32_t max_idx,
                                   resize_nearest_mode_t nearest_mode) {
  int32_t in_idx;
  switch (nearest_mode) {
  case RESIZE_NEAREST_CEIL:
    in_idx = (int32_t)ceilf(x);
    break;
  case RESIZE_NEAREST_ROUND_PREFER_FLOOR:
    /* At exactly n+0.5 choose floor; otherwise standard round. */
    in_idx = (x - floorf(x) == 0.5f) ? (int32_t)floorf(x) : (int32_t)roundf(x);
    break;
  case RESIZE_NEAREST_ROUND_PREFER_CEIL:
    in_idx = (int32_t)roundf(x);
    break;
  default: /* RESIZE_NEAREST_FLOOR */
    in_idx = (int32_t)floorf(x);
    break;
  }
  return CLAMP(in_idx, 0, max_idx);
}

void Resize(const void *input, void *output, resize_type_t type_tag, int32_t N,
            int32_t C, int32_t spatial_dims, const int32_t *input_shape,
            const int32_t *output_shape, resize_mode_t mode,
            resize_coord_mode_t coord_mode,
            resize_nearest_mode_t nearest_mode) {

  if (N <= 0 || C <= 0 || spatial_dims < 1 ||
      spatial_dims > RESIZE_MAX_SPATIAL_DIMS)
    return;

  /* not implemented */
  if (mode == RESIZE_MODE_CUBIC ||
      coord_mode == RESIZE_COORD_TF_CROP_AND_RESIZE)
    return;

  uint32_t elem_size = _resize_element_size(type_tag);

  /* Row-major strides for the spatial dimensions. */
  int32_t out_strides[RESIZE_MAX_SPATIAL_DIMS];
  int32_t in_strides[RESIZE_MAX_SPATIAL_DIMS];
  int32_t L_out = 1, L_in = 1;
  out_strides[spatial_dims - 1] = 1;
  in_strides[spatial_dims - 1] = 1;
  for (int32_t d = spatial_dims - 2; d >= 0; d--) {
    out_strides[d] = out_strides[d + 1] * output_shape[d + 1];
    in_strides[d] = in_strides[d + 1] * input_shape[d + 1];
  }
  for (int32_t d = 0; d < spatial_dims; d++) {
    L_out *= output_shape[d];
    L_in *= input_shape[d];
  }

  for (int32_t n = 0; n < N; n++) {
    for (int32_t c = 0; c < C; c++) {
      int32_t in_base = (n * C + c) * L_in;
      int32_t out_base = (n * C + c) * L_out;

      for (int32_t oi = 0; oi < L_out; oi++) {
        int32_t rem = oi;

        if (mode == RESIZE_MODE_NEAREST) {
          /* Nearest-neighbour: map each spatial coord and copy the element. */
          int32_t in_flat = 0;
          for (int32_t d = 0; d < spatial_dims; d++) {
            int32_t out_idx = rem / out_strides[d];
            rem -= out_idx * out_strides[d];
            float32_t x = _resize_get_coord(out_idx, input_shape[d],
                                            output_shape[d], coord_mode);
            in_flat +=
                _resize_nearest_idx(x, input_shape[d] - 1, nearest_mode) *
                in_strides[d];
          }
          memcpy((char *)output + (uint32_t)(out_base + oi) * elem_size,
                 (const char *)input +
                     (uint32_t)(in_base + in_flat) * elem_size,
                 elem_size);

        } else {

          float32_t x_in[RESIZE_MAX_SPATIAL_DIMS]; // fractional input coord
          int32_t lo[RESIZE_MAX_SPATIAL_DIMS];     // index lower than x_in
          int32_t hi[RESIZE_MAX_SPATIAL_DIMS];     // index higher than x_in
          float32_t w_lo[RESIZE_MAX_SPATIAL_DIMS]; // low interpolation weight
          float32_t w_hi[RESIZE_MAX_SPATIAL_DIMS]; // high interpolation weight

          /*
          prepares the data for the N-linear interpolation that follows.
          For each spatial dimension d:
          - Extract the per-dimension output coordinate from the flat index oi
          - Map the output coordinate to a fractional input coordinate x_in
          - Find the two bracketing input indices (lo, hi) along dimension d
          - Compute interpolation weights (w_hi, w_lo)
          */
          for (int32_t d = 0; d < spatial_dims; d++) {
            int32_t out_idx = rem / out_strides[d];
            rem -= out_idx * out_strides[d]; // flat output index
            x_in[d] = _resize_get_coord(out_idx, input_shape[d],
                                        output_shape[d], coord_mode);
            x_in[d] = CLAMP(x_in[d], 0.0f, (float32_t)(input_shape[d] - 1));
            lo[d] = (int32_t)floorf(x_in[d]);
            hi[d] = (lo[d] + 1 < input_shape[d]) ? lo[d] + 1 : lo[d];
            w_hi[d] = x_in[d] - (float32_t)lo[d];
            w_lo[d] = 1.0f - w_hi[d];
          }

          /*
          N-linear interpolation: weighted sum over the 2^spatial_dims corners.

          example: spatial_dims = 2 (bilinear), there are 2^2 = 4 corners
          corner=0 (bits: 00) → (lo[0], lo[1])   weight = w_lo[0] * w_lo[1]
          corner=1 (bits: 01) → (hi[0], lo[1])   weight = w_hi[0] * w_lo[1]
          corner=2 (bits: 10) → (lo[0], hi[1])   weight = w_lo[0] * w_hi[1]
          corner=3 (bits: 11) → (hi[0], hi[1])   weight = w_hi[0] * w_hi[1]
          */
          float32_t result = 0.0f;
          for (int32_t corner = 0, n_corners = 1 << spatial_dims;
               corner < n_corners; corner++) {
            float32_t weight = 1.0f;
            int32_t in_flat = 0;
            for (int32_t d = 0; d < spatial_dims; d++) {
              if ((corner >> d) & 1) {
                weight *= w_hi[d];
                in_flat += hi[d] * in_strides[d];
              } else {
                weight *= w_lo[d];
                in_flat += lo[d] * in_strides[d];
              }
            }
            result += weight * _resize_read(input, in_base + in_flat, type_tag);
          }
          _resize_write(output, out_base + oi, result, type_tag);
        }
      }
    }
  }
}
