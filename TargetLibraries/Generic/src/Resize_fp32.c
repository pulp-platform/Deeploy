#include "DeeployBasicMath.h"
#include <math.h>

void Resize_fp32_fp32(float32_t *data_in, int32_t *inputA, int32_t shapeA_len,
                      float32_t *scale, int32_t scale_len, float32_t *data_out,
                      int8_t coord, int8_t mode, int8_t nearest_mode) {
  // we will code only the version we need :
  // coordinate_transformation : asymmetric (4)
  // mode : nearest(default 0)
  // nearest_mode : floor (2) the difference between 0.5 scales outputing [0]
  // and [2] vs [1] and [3]

  // X : input
  // roi : empty
  // scales : multiplies x to get output shape. 3*0.66=1.98 ==1 always floors
  int32_t max_len = MAX(shapeA_len, scale_len); // they should always be equal
  int32_t shape1[max_len];

  // prepend the "1"s
  for (int i = 0; i < max_len; i++) {
    if (i < max_len - shapeA_len) {
      shape1[i] = 1;
    } else {
      shape1[i] = inputA[i - max_len + shapeA_len];
    }
  }

  // prepare the product length
  int32_t prod1[max_len];
  int32_t mul1 = 1;
  for (int32_t i = max_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;
    mul1 = mul1 * shape1[i];
  }

  // shape_out
  int32_t shape_out[max_len];
  int32_t mul_out = 1;
  // mul out doubles as size of output
  for (int32_t i = 0; i < max_len; i++) {
    shape_out[i] =
        (int32_t)(scale[i] * shape1[i]); // LES PARENTHESES SONT IMPORTANTES ICI
    mul_out = (mul_out)*shape_out[i];
  }

  int32_t posfinalA;
  int32_t posfinalB;

  int32_t quotients[max_len];
  for (int32_t i = 0; i < max_len; i++) {
    quotients[i] = 0;
  }

  for (int32_t i = 0; i < mul_out;
       i++) { // COULD BE SMALLER THAN shape1, thus requiring the nearest_mode
    posfinalA = 0;
    posfinalB = 0;

    // Mixed Radix Conversion to get the indices, but we know the previous
    // result
    for (int32_t dim = max_len - 1; dim >= 0;
         dim--) { // which i /quotients values are the ones also in A? the
                  // corners?
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
        posfinalA =
            posfinalA + prod1[dim] * (int32_t)(quotients[dim] / scale[dim]);
      } // else posfinalA+=quotients[dim]*0
    }
    data_out[i] = data_in[posfinalA];
    quotients[max_len - 1] += 1;
  }
}
