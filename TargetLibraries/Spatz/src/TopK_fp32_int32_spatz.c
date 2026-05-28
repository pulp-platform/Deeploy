#include "DeeployBasicMath.h"
#include <math.h>

void compute_topk_min_heap(float32_t *data_in,
                           float32_t *values_out,
                           int32_t *indices_out,
                           uint32_t k,
                           uint32_t n,
                           float32_t *heap_values,
                           int32_t *heap_indices) {
  // Initialize heap with first k elements
  for (uint32_t i = 0; i < k; ++i) {
    heap_values[i] = data_in[i];
    heap_indices[i] = (int32_t)i;
  }

  // Build min-heap
  for (int32_t root = (int32_t)k / 2 - 1; root >= 0; --root) {
    uint32_t idx = (uint32_t)root;
    for (;;) {
      uint32_t left = 2 * idx + 1;
      if (left >= k) {
        break;
      }
      uint32_t smallest = left;
      uint32_t right = left + 1;
      if (right < k && heap_values[right] < heap_values[left]) {
        smallest = right;
      }
      if (heap_values[smallest] < heap_values[idx]) {
        float32_t tmp_val = heap_values[idx];
        int32_t tmp_idx = heap_indices[idx];
        heap_values[idx] = heap_values[smallest];
        heap_indices[idx] = heap_indices[smallest];
        heap_values[smallest] = tmp_val;
        heap_indices[smallest] = tmp_idx;
        idx = smallest;
      } else {
        break;
      }
    }
  }

  // Process remaining elements, keeping top k values in the min-heap
  for (uint32_t i = k; i < n; ++i) {
    float32_t value = data_in[i];
    if (value > heap_values[0]) {
      heap_values[0] = value;
      heap_indices[0] = (int32_t)i;

      uint32_t idx = 0;
      for (;;) {
        uint32_t left = 2 * idx + 1;
        if (left >= k) {
          break;
        }
        uint32_t smallest = left;
        uint32_t right = left + 1;
        if (right < k && heap_values[right] < heap_values[left]) {
          smallest = right;
        }
        if (heap_values[smallest] < heap_values[idx]) {
          float32_t tmp_val = heap_values[idx];
          int32_t tmp_idx = heap_indices[idx];
          heap_values[idx] = heap_values[smallest];
          heap_indices[idx] = heap_indices[smallest];
          heap_values[smallest] = tmp_val;
          heap_indices[smallest] = tmp_idx;
          idx = smallest;
        } else {
          break;
        }
      }
    }
  }

  // Sort the final top-k values in descending order for output
  for (uint32_t i = 0; i < k; ++i) {
    uint32_t max_idx = i;
    for (uint32_t j = i + 1; j < k; ++j) {
      if (heap_values[j] > heap_values[max_idx]) {
        max_idx = j;
      }
    }
    if (max_idx != i) {
      float32_t tmp_val = heap_values[i];
      int32_t tmp_idx = heap_indices[i];
      heap_values[i] = heap_values[max_idx];
      heap_indices[i] = heap_indices[max_idx];
      heap_values[max_idx] = tmp_val;
      heap_indices[max_idx] = tmp_idx;
    }
    values_out[i] = heap_values[i];
    indices_out[i] = heap_indices[i];
  }
}