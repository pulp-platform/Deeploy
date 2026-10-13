#include "DeeployBasicMath.h"

void Where_fp32_fp32(const int8_t *__restrict__ pSrcCond,
                     int32_t offset, // don't know how to create BOOL_T datatype
                     const float32_t *__restrict__ pSrcX,
                     const float32_t *__restrict__ pSrcY, int32_t *inputCond,
                     int32_t shapeCond_len, int32_t *inputX, int32_t shapeX_len,
                     int32_t *inputY, int32_t shapeY_len,
                     float32_t *__restrict__ pDstC) {

  int32_t max_len = MAX(shapeCond_len, MAX(shapeX_len, shapeY_len));
  int32_t shape1[max_len];
  int32_t shape2[max_len];
  int32_t shape3[max_len];

  // prepare the product length
  int32_t prod1[max_len];
  int32_t prod2[max_len];
  int32_t prod3[max_len];

  // shape_out
  int32_t shape_out[max_len];
  int32_t mul_out = 1;
  // mul out doubles as size of output

  // prepend the "1"s
  for (int i = 0; i < max_len; i++) {
    if (i < max_len - shapeCond_len) {
      shape1[i] = 1;
    } else {
      shape1[i] = inputCond[i - max_len + shapeCond_len];
    }
  }
  for (int i = 0; i < max_len; i++) {
    if (i < max_len - shapeX_len) {
      shape2[i] = 1;
    } else {
      shape2[i] = inputX[i - max_len + shapeX_len];
    }
  }
  for (int i = 0; i < max_len; i++) {
    if (i < max_len - shapeY_len) {
      shape3[i] = 1;
    } else {
      shape3[i] = inputY[i - max_len + shapeY_len];
    }
  }

  // prepare the product length
  int32_t mul1 = 1;
  for (int32_t i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;
    mul1 = mul1 * shape1[i];
  }
  int32_t mul2 = 1;
  for (int32_t i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod2[i] = mul2;
    mul2 = mul2 * shape2[i];
  }
  int32_t mul3 = 1;
  for (int32_t i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod3[i] = mul3;
    mul3 = mul3 * shape3[i];
  }

  // shape_out
  for (int32_t i = 0; i < max_len; i++) {
    shape_out[i] = MAX(shape3[i], MAX(shape1[i], shape2[i]));
    mul_out = (mul_out)*shape_out[i];
  }
  // printf("mul out %d\n", mul_out);
  int32_t posfinalCond, posfinalX, posfinalY;

  int32_t quotients[max_len];
  for (int32_t i = 0; i < max_len; i++) {
    quotients[i] = 0;
  }
  for (int32_t i = 0; i < mul_out; i++) {
    posfinalCond = 0;
    posfinalX = 0;
    posfinalY = 0;

    // Mixed Radix Conversion to get the indices, but we know the previous
    // result
    for (int32_t dim = max_len - 1; dim >= 0; dim--) {
      if (quotients[dim] >= shape_out[dim]) {
        quotients[dim] = 0;
        quotients[dim - 1] +=
            1; // should always stop before "quotients[0]==shape_out[0]" -->
               // "quotients[-1]"
      } else {
        break; // quotients are all correctly updated
      }
    }
    for (int32_t dim = 0; dim < max_len; dim++) {
      if (shape1[dim] != 1) {
        posfinalCond = posfinalCond + quotients[dim] * prod1[dim];
      } // else posfinalCond+=quotients[dim]*0
      if (shape2[dim] != 1) {
        posfinalX = posfinalX + quotients[dim] * prod2[dim];
      } // else posfinalX+=quotients[dim]*0
      if (shape3[dim] != 1) {
        posfinalY = posfinalY + quotients[dim] * prod3[dim];
      } // else posfinalY+=quotients[dim]*0
    }
    // triple broadcasting
    if (pSrcCond[posfinalCond] + offset) {
      pDstC[i] = pSrcX[posfinalX];
    } else {
      pDstC[i] = pSrcY[posfinalY];
    }
    // pDstC[i]= (pSrcCond[posfinalCond]+128) ? pSrcX[posfinalX] :
    // pSrcY[posfinalY];//does not work because???
    quotients[max_len - 1] += 1;
  }
}
