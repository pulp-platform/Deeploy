#include "DeeployBasicMath.h"

void GatherElements_fp32_s64(const float32_t *__restrict__ pSrcA,
                             int32_t *inputA, int32_t shapeA_len,
                             const int64_t *__restrict__ pSrcB, int32_t *inputB,
                             int32_t shapeB_len, int64_t offset, int32_t axis,
                             float32_t *__restrict__ pDstC) {

  // The old Template
  int32_t max_len = MAX(shapeA_len, shapeB_len);
  int32_t shape1[max_len];
  int32_t shape2[max_len];

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
      shape2[i] = inputB[i - max_len + shapeB_len];
    }
  }
  // endofTemplate

  // prepare the product length
  uint32_t prod1[max_len];
  uint32_t prod2[max_len];

  uint32_t mul1 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;
    mul1 = mul1 * shape1[i];
  }
  uint32_t mul2 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod2[i] = mul2;
    mul2 = mul2 * shape2[i];
  }
  // shape_out IS shape2

  uint32_t posfinalA;

  // positive axis
  int32_t current_axis = axis;
  if (0 > current_axis) {
    current_axis = current_axis + shapeA_len;
  }

  uint32_t quotients[max_len];
  for (uint32_t i = 0; i < max_len; i++) {
    quotients[i] = 0;
  }
  for (uint32_t i = 0; i < mul2; i++) {
    posfinalA = 0;

    // Mixed Radix Conversion to get the indices, but we know the previous
    // number's result

    // quotients[-1]+=1;
    // if quotients[-1]==shape_out[-1]
    //   quotients[-1]=0;
    //   quotients[-2]+=1;
    //   if quotients[-2]==shape_out[-2]
    //       quotients[-2]=0
    //       ...
    for (int32_t dim = max_len - 1; dim >= 0; dim--) {
      if (quotients[dim] >= shape2[dim]) { // shape_out
        quotients[dim] = 0;
        quotients[dim - 1] +=
            1; // should always stop before "quotients[0]==shape_out[0]" -->
               // "quotients[-1]"
      } else {
        break; // quotients are all correctly updated
      }
    }
    // select axis
    for (int32_t dim = 0; dim < max_len; dim++) {
      if (dim == current_axis) {
        posfinalA = posfinalA + (pSrcB[i] + offset) * prod1[dim];
      } else {
        posfinalA = posfinalA + quotients[dim] * prod1[dim];
      }
    }
    pDstC[i] = pSrcA[posfinalA];
    quotients[max_len - 1] += 1;
  }
}

void GatherElements_fp32_s32(const float32_t *__restrict__ pSrcA,
                             int32_t *inputA, int32_t shapeA_len,
                             const int32_t *__restrict__ pSrcB, int32_t *inputB,
                             int32_t shapeB_len, int32_t offset, int32_t axis,
                             float32_t *__restrict__ pDstC) {

  // The old Template
  int32_t max_len = MAX(shapeA_len, shapeB_len);
  int32_t shape1[max_len];
  int32_t shape2[max_len];

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
      shape2[i] = inputB[i - max_len + shapeB_len];
    }
  }
  // endofTemplate

  // prepare the product length
  uint32_t prod1[max_len];
  uint32_t prod2[max_len];

  uint32_t mul1 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;
    mul1 = mul1 * shape1[i];
  }
  uint32_t mul2 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod2[i] = mul2;
    mul2 = mul2 * shape2[i];
  }
  // shape_out IS shape2

  uint32_t posfinalA;

  // positive axis
  int32_t current_axis = axis;
  if (0 > current_axis) {
    current_axis = current_axis + shapeA_len;
  }

  uint32_t quotients[max_len];
  for (uint32_t i = 0; i < max_len; i++) {
    quotients[i] = 0;
  }
  for (uint32_t i = 0; i < mul2; i++) {
    posfinalA = 0;

    // Mixed Radix Conversion to get the indices, but we know the previous
    // number's result

    // quotients[-1]+=1;
    // if quotients[-1]==shape_out[-1]
    //   quotients[-1]=0;
    //   quotients[-2]+=1;
    //   if quotients[-2]==shape_out[-2]
    //       quotients[-2]=0
    //       ...
    for (int32_t dim = max_len - 1; dim >= 0; dim--) {
      if (quotients[dim] >= shape2[dim]) { // shape_out
        quotients[dim] = 0;
        quotients[dim - 1] +=
            1; // should always stop before "quotients[0]==shape_out[0]" -->
               // "quotients[-1]"
      } else {
        break; // quotients are all correctly updated
      }
    }
    // select axis
    for (int32_t dim = 0; dim < max_len; dim++) {
      if (dim == current_axis) {
        posfinalA = posfinalA + (pSrcB[i] + offset) * prod1[dim];
      } else {
        posfinalA = posfinalA + quotients[dim] * prod1[dim];
      }
    }
    pDstC[i] = pSrcA[posfinalA];
    quotients[max_len - 1] += 1;
  }
}
