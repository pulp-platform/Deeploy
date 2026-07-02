#include "DeeployBasicMath.h"
#include <math.h>
#include <float.h>

/* note:
 * heap is stored in a vector
 * minimum element is in root of heap (index 0 in the vector)
 * left and right of a index are always > than root
 */
static inline __attribute__((always_inline)) void reorder_heap(uint32_t idx, uint32_t size, float32_t *heap_values, int32_t *heap_indices){
  for (;;) {
    uint32_t left = 2 * idx + 1;
    if (left >= size) {
      break;
    }
    uint32_t smallest = left;
    uint32_t right = left + 1;
    if (right < size && heap_values[right] < heap_values[left]) {
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

// heap_value and _indices are arrays i can modify and work with, used as scratchpad, but also as output
void compute_topk_min_heap( uint32_t k, uint32_t n, float32_t *data_in, float32_t *heap_values, int32_t *heap_indices) {
  // Initialize heap with first k elements
  for (uint32_t i = 0; i < k; ++i) { heap_values[i] = data_in[i]; heap_indices[i] = (int32_t)i; }

  // Build min-heap by reordeing each sub heap starting fomr the smallest ones (k/2-1) to the biggest ones (0)
  for (int32_t root = (int32_t)k / 2 - 1; root >= 0; --root) {
    reorder_heap(root, k, heap_values, heap_indices);
  }

  // Process remaining elements, keeping top k values in the min-heap
  for (uint32_t i = k; i < n; ++i) {
    float32_t value = data_in[i];
    if (value > heap_values[0]) {
      heap_values[0] = value;
      heap_indices[0] = (int32_t)i;

      reorder_heap(0, k, heap_values, heap_indices);
    }
  }

  /* heap sort */
  for (uint32_t i = k-1; i > 0; i--) {
    // swap min and max, root and most bottom (biggest) leaf
    float32_t root_val = heap_values[0]; float32_t root_idx = heap_indices[0];

    heap_values[0] = heap_values[i]; heap_indices[0] = heap_indices[i];

    heap_values[i] = root_val; heap_indices[i] = root_idx;
    // reduce size and heapify
    reorder_heap(0, i, heap_values, heap_indices);
  }
  
}

// finds the k biggest elements from a vector of n elements, and returns them in data_out
void compute_topk_vector_instructions(uint32_t k, uint32_t n, float32_t *data_in, float32_t *data_out, int32_t *indices_out) {
    
    for (uint32_t i = 0; i < k; i++) {
        float32_t global_max = -FLT_MAX;
        int32_t global_max_idx = -1;
        
        uint32_t avl = n;
        uint32_t vl;
        float32_t *ptr = data_in;
        uint32_t current_idx_offset = 0;

        // --- Pass 1: Find the maximum value and its index in the current array ---
        while (avl > 0) {
            asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(avl));

            // Setup scalar helper registers for reduction initialization
            float32_t block_max_scalar = -FLT_MAX;
            
            // Inline assembly to load, reduce, and find the index manually or via step tracking
            // v24 will hold the loaded data chunks
            asm volatile (
                "vle32.v v24, (%1)\n\t"
                "vfmv.s.f v0, %2\n\t"              // Init scalar reduction register with -FLT_MAX
                "vfredmax.vs v0, v24, v0\n\t"       // Find max in this vector block
                "vfmv.f.s %0, v0\n\t"               // Move block max back to C variable
                : "=f"(block_max_scalar)
                : "r"(ptr), "f"(-FLT_MAX)
                : "v0", "v24"
            );

            // Check if the maximum found in this block beats our global tracker
            if (block_max_scalar > global_max) {
                // If it does, we sweep the block to catch the exact scalar index position
                for (uint32_t j = 0; j < vl; j++) {
                    if (ptr[j] > global_max) {
                        global_max = ptr[j];
                        global_max_idx = current_idx_offset + j;
                    }
                }
            }

            ptr += vl;
            current_idx_offset += vl;
            avl -= vl;
        }

        // Save the found top element metadata to output arrays
        data_out[i] = global_max;
        indices_out[i] = global_max_idx;

        // --- Pass 2: Mask out the found maximum to prevent re-discovery ---
        if (global_max_idx != -1) {
            data_in[global_max_idx] = -FLT_MAX;
        }
    }
}