#include "DeeployBasicMath.h"

// insertion sort https://www.geeksforgeeks.org/dsa/insertion-sort-algorithm/
void insertion_sort(float32_t *values, int64_t *indices, int32_t len) {

  for (int32_t i = 1; i < len; ++i) {
    float32_t key = values[i];
    int64_t id = indices[i];

    int32_t j = i - 1;

    /* Move elements of arr[0..i-1], that are
       smaller than key, to one position ahead
       of their current position */
    while (j >= 0 && values[j] < key) {
      values[j + 1] = values[j];
      indices[j + 1] = indices[j];
      j = j - 1;
    }
    values[j + 1] = key;
    indices[j + 1] = id;
  }
}

void TopK_fp32(const float32_t *__restrict__ pSrcA,
               float32_t *__restrict__ pDstC, int64_t *__restrict__ pDstI,
               int64_t I_offset, // indices, maybe no offset
               const int64_t *__restrict__ K, int64_t K_offset, int32_t in_axis,
               int32_t *inputA, int32_t shapeA_len) {

  // The old Template
  int32_t max_len = shapeA_len;
  int32_t shape1[max_len];
  int32_t shape2[max_len];

  // int32_t inputA[shapeA_len];
  // int32_t inputB[shapeB_len];

  // prepend the "1"s
  for (int i = 0; i < max_len; i++) {
    if (i < max_len - shapeA_len) {
      shape1[i] = 1;
    } else {
      shape1[i] = inputA[i - max_len + shapeA_len];
    }
  }
  // endofTemplate

  // prepare the product length
  uint32_t prod1[max_len];

  uint32_t mul1 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;
    mul1 = mul1 * shape1[i];
  }
  int32_t axis = (in_axis + shapeA_len) % shapeA_len;
  // shape_out
  uint32_t shape_out[max_len];
  uint32_t mul_out = 1;
  for (int i = 0; i < max_len; i++) {
    if (i == axis) {
      shape_out[i] = (*K + K_offset);
    } else {
      shape_out[i] = shape1[i];
    }
  }
  // mul out doubles as size of output
  uint32_t prod_out[max_len];
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod_out[i] = mul_out;
    mul_out = mul_out * shape_out[i];
  }

  // shape_target for parcouring every dim except axis
  uint32_t shape_target[max_len];
  uint32_t mul_target = 1;
  for (int i = 0; i < max_len; i++) {
    if (i == axis) {
      shape_target[i] = 1;
    } else {
      shape_target[i] = shape1[i];
    }
    mul_target = mul_target * shape_target[i];
  }

  uint32_t posfinalA, posfinalC;

  uint32_t quotients[max_len];
  for (uint32_t i = 0; i < max_len; i++) {
    quotients[i] = 0;
  }

  float32_t candidates[shape1[axis]]; // shape out axis should be K, do
                                      // cadidates is shape_in[axis]
  int64_t indices[shape1[axis]];
  for (uint32_t i = 0; i < mul_target;
       i++) { // every element excluding the axis=dim
    posfinalA = 0;
    posfinalC = 0;

    for (int32_t dim = max_len - 1; dim >= 0; dim--) {
      if (quotients[dim] >=
          shape_target[dim]) { // shape target to ignore axis=dim,
                               // quotients[axis] always 0
        quotients[dim] = 0;
        quotients[dim - 1] += 1;
      } else {
        break;
      }
    }
    for (int32_t dim = 0; dim < max_len; dim++) {
      posfinalA =
          posfinalA +
          quotients[dim] * prod1[dim]; // for axis=dim : posfinal+0 * prod1
      posfinalC = posfinalC + quotients[dim] * prod_out[dim];
    } // multidim position without taking into account the K dimension

    // get all candidates in the axis of K
    for (int64_t a = 0; a < shape1[axis]; a++) {
      candidates[a] =
          pSrcA[posfinalA + a * prod1[axis]]; // same quotients, but gets K
                                              // elements along the axis=dim
      indices[a] = a;
      // we want the values AND the indices sorted : candidates[a] and a
    }
    // sort
    insertion_sort(candidates, indices, shape1[axis]);
    // printf("Pos %d Vals %f \t %f", posfinalA, candidates[0], candidates[1] );

    // get K kargest
    for (int32_t l = 0; l < (*K) + K_offset; l++) {
      pDstC[posfinalC + l * prod_out[axis]] = candidates[l];
      pDstI[posfinalC + l * prod_out[axis]] = indices[l] + I_offset;
    }
    quotients[max_len - 1] += 1;
  }
}
