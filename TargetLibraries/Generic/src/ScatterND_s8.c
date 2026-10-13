#include "DeeployBasicMath.h"

void ScatterND_s8(const int8_t *__restrict__ data,
                  const int64_t *__restrict__ indices,
                  int32_t indices_offset_deeploy,
                  const int8_t *__restrict__ updates,
                  int8_t *__restrict__ output, const int64_t *__restrict__ d,
                  int32_t n_idx, int32_t n_kr, int32_t k, int32_t data_size) {
  const int64_t *kt = indices;
  const int8_t *updates_pt = updates;
  int8_t *output_slice_pt;
  int64_t block_size;

  // copy data to output
  for (uint32_t i = 0; i < data_size; i++) {
    output[i] = data[i];
  }

  // apply updates
  for (uint32_t p = 0; p < n_idx; p++) {

    block_size = n_kr;
    output_slice_pt =
        output + (int64_t)(kt[k - 1] + indices_offset_deeploy) * block_size;

    for (uint32_t j = 1; j < k; j++) {
      block_size *= (int64_t)d[k - j];
      output_slice_pt +=
          (int64_t)(kt[k - 1 - j] + indices_offset_deeploy) * block_size;
    }

    for (uint32_t i = 0; i < n_kr; i++) {
      output_slice_pt[i] = *updates_pt++;
    }
    kt += k;
  }
}
