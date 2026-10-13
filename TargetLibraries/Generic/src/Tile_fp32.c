#include "DeeployBasicMath.h"

void Tile_fp32(const float32_t *__restrict__ pSrcA,
               float32_t *__restrict__ pDstC, const int64_t *__restrict__ pSrcB,
               int32_t shapeB_len, int64_t offset, int32_t *inputA,
               int32_t shapeA_len) {

  // The old Template
  int32_t max_len = MAX(shapeA_len, shapeB_len);
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
  for (int i = 0; i < max_len; i++) {
    if (i < max_len - shapeB_len) {
      shape2[i] = 1;
    } else {
      shape2[i] = pSrcB[i - max_len + shapeB_len] + offset;
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

  // shape_out
  uint32_t shape_out[max_len];
  uint32_t mul_out = 1;
  for (int i = 0; i < max_len; i++) {
    shape_out[i] = shape1[i] * shape2[i]; // important
    mul_out = mul_out * shape_out[i];
  }
  // mul out doubles as size of output

  uint32_t posfinalA;

  uint32_t quotients[max_len];
  for (uint32_t i = 0; i < max_len; i++) {
    quotients[i] = 0;
  }

  for (uint32_t i = 0; i < mul_out; i++) {
    posfinalA = 0;
    for (int32_t dim = max_len - 1; dim >= 0; dim--) {
      if (quotients[dim] >= shape_out[dim]) { // maybe shape1 instead
        quotients[dim] = 0;
        quotients[dim - 1] +=
            1; // should always stop before "quotients[0]==shape_out[0]" -->
               // "quotients[-1]"
      } else {
        break; // quotients are all correctly updated
      }
    }
    for (int32_t dim = 0; dim < max_len; dim++) {
      posfinalA = posfinalA + (quotients[dim] % shape1[dim]) * prod1[dim];
    }
    pDstC[i] = pSrcA[posfinalA];
    quotients[max_len - 1] += 1;
  }
}
