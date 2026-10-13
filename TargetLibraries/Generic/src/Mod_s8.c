#include "DeeployBasicMath.h"

void Mod_s8_s8(const int8_t *__restrict__ pSrcA, int8_t A_offset,
               const int8_t *__restrict__ pSrcB, int8_t B_offset, int8_t fmod,
               int8_t *__restrict__ pDstC, int8_t C_offset, int32_t *inputA,
               int32_t shapeA_len, int32_t *inputB, int32_t shapeB_len) {

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

  // prepare the product length
  uint32_t prod1[max_len];
  uint32_t prod2[max_len];

  uint32_t mul1 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;                       // 1 2 6 24 48
    mul1 = mul1 * shape1[i];
  }
  uint32_t mul2 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod2[i] = mul2;                       // 1 2 6
    mul2 = mul2 * shape2[i];
  }

  // shape_out
  uint32_t shape_out[max_len];
  for (int i = 0; i < max_len; i++) {
    shape_out[i] = MAX(shape1[i], shape2[i]);
    // printf("max %d and %d gives  %d\n",shape1[i], shape2[i], shape_out[i]);
  }

  uint32_t mul_out = 1;
  uint32_t prod_out[max_len];
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod_out[i] = mul_out;                 // 1 2 6
    mul_out = mul_out * shape_out[i];
  }
  // mul out doubles as size of output
  uint32_t posfinalA;
  uint32_t posfinalB;

  int8_t a = 0;
  int8_t b = 0;

  for (uint32_t i = 0; i < mul_out; i++) { // iterate over every element
    uint32_t rem = i;
    uint32_t quotient = 0;
    posfinalA = 0;
    posfinalB = 0;
    for (uint32_t dim = 0; dim < max_len;
         dim++) { // probably a smarter way with modulos
      // printf("divider %d     ", prod_out[dim]);
      quotient = rem / prod_out[dim];
      rem = rem % prod_out[dim];

      if (shape1[dim] == 1) {
        // posA[dim]=0;
        posfinalA = posfinalA + 0;
      } else {
        // posA[dim]=quotient;
        posfinalA = posfinalA + quotient * prod1[dim];
      }
      if (shape2[dim] == 1) {
        // posB[dim]=0;
        posfinalB = posfinalB + 0;
      } else {
        // posB[dim]=quotient;
        posfinalB = posfinalB + quotient * prod2[dim];
      }
      // posfinalA=posfinalA+posA[dim]*prod1[dim];
      // posfinalB=posfinalB+posB[dim]*prod2[dim];
    }
    if (fmod == 0) {
      // https://stackoverflow.com/questions/828092/python-style-integer-division-modulus-in-c

      // pDstC[i]=pSrcA[posfinalA]%pSrcB[posfinalB];//NEEDS to be Python style
      // modulo
      a = pSrcA[posfinalA] + A_offset;
      b = pSrcB[posfinalB] + B_offset;
      if (a < 0)
        if (b < 0)
          pDstC[i] = -(-a % -b) + C_offset;
        else
          pDstC[i] = -a % b + C_offset - (-a % -b != 0 ? 1 : 0);
      else if (b < 0)
        pDstC[i] = -(a % -b) + C_offset + (-a % -b != 0 ? 1 : 0);
      else
        pDstC[i] = a % b + C_offset;
    }
  }
}

void Mod_s64_s64(const int64_t *__restrict__ pSrcA, int64_t A_offset,
                 const int64_t *__restrict__ pSrcB, int64_t B_offset,
                 int8_t fmod, int64_t *__restrict__ pDstC, int64_t C_offset,
                 int32_t *inputA, int32_t shapeA_len, int32_t *inputB,
                 int32_t shapeB_len) {

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

  // prepare the product length
  uint32_t prod1[max_len];
  uint32_t prod2[max_len];

  uint32_t mul1 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;                       // 1 2 6 24 48
    mul1 = mul1 * shape1[i];
  }
  uint32_t mul2 = 1;
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod2[i] = mul2;                       // 1 2 6
    mul2 = mul2 * shape2[i];
  }

  // shape_out
  uint32_t shape_out[max_len];
  for (int i = 0; i < max_len; i++) {
    shape_out[i] = MAX(shape1[i], shape2[i]);
    // printf("max %d and %d gives  %d\n",shape1[i], shape2[i], shape_out[i]);
  }

  uint32_t mul_out = 1;
  uint32_t prod_out[max_len];
  for (int i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod_out[i] = mul_out;                 // 1 2 6
    mul_out = mul_out * shape_out[i];
  }
  // mul out doubles as size of output
  uint32_t posfinalA;
  uint32_t posfinalB;

  int64_t a = 0;
  int64_t b = 0;

  for (uint32_t i = 0; i < mul_out; i++) { // iterate over every element
    uint32_t rem = i;
    uint32_t quotient = 0;
    posfinalA = 0;
    posfinalB = 0;
    for (uint32_t dim = 0; dim < max_len;
         dim++) { // probably a smarter way with modulos
      // printf("divider %d     ", prod_out[dim]);
      quotient = rem / prod_out[dim];
      rem = rem % prod_out[dim];

      if (shape1[dim] == 1) {
        // posA[dim]=0;
        posfinalA = posfinalA + 0;
      } else {
        // posA[dim]=quotient;
        posfinalA = posfinalA + quotient * prod1[dim];
      }
      if (shape2[dim] == 1) {
        // posB[dim]=0;
        posfinalB = posfinalB + 0;
      } else {
        // posB[dim]=quotient;
        posfinalB = posfinalB + quotient * prod2[dim];
      }
      // posfinalA=posfinalA+posA[dim]*prod1[dim];
      // posfinalB=posfinalB+posB[dim]*prod2[dim];
    }
    if (fmod == 0) {

      // pDstC[i]=pSrcA[posfinalA]%pSrcB[posfinalB];//NEEDS to be Python style
      // modulo
      a = pSrcA[posfinalA] + A_offset;
      b = pSrcB[posfinalB] + B_offset;

      // https://stackoverflow.com/questions/1907565/c-and-python-different-behaviour-of-the-modulo-operation
      pDstC[i] = ((a % b) + b) % b + C_offset;
      /*
      //https://stackoverflow.com/questions/828092/python-style-integer-division-modulus-in-c
      if (a < 0)
          if (b < 0)
              pDstC[i]= -(-a % -b)+C_offset;
          else
              pDstC[i]= -a % b +C_offset- (-a % -b != 0 ? 1 : 0);
      else if (b < 0)
              pDstC[i]= -(a % -b) +C_offset+ (-a % -b != 0 ? 1 : 0);
          else
              pDstC[i]= a % b + C_offset;*/
    }
  }
}
