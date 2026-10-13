#include "DeeployBasicMath.h"

void Split_fp32(const float32_t *__restrict__ pSrcA, int32_t *inputA,
                int32_t shapeA_len, const int64_t *__restrict__ pSrcB,
                int32_t sizeB, int64_t offset, int32_t axis_in,
                float32_t **__restrict__ pDstC, int32_t num_output) {

  // prepare the product length
  uint32_t prod1[shapeA_len];

  uint32_t mul1 = 1;
  for (int i = shapeA_len - 1; i >= 0; i--) { // 3 2 1 0
    prod1[i] = mul1;
    mul1 = mul1 * inputA[i];
  }
  // mul doubles as size of output

  // get the targeted dim
  int32_t axis = (axis_in + shapeA_len) % shapeA_len;

  uint32_t target = 0;
  uint32_t pos[num_output];
  for (uint32_t i = 0; i < num_output; i++) {
    pos[i] = 0;
  }

  uint32_t quotients[shapeA_len];
  for (uint32_t i = 0; i < shapeA_len; i++) {
    quotients[i] = 0;
  }
  for (uint32_t i = 0; i < mul1; i++) {
    // printf("We are at %d\n",i);

    // Mixed Radix Conversion to get the indices, but we know the previous
    // number's result

    // quotients[-1]+=1;
    // if quotients[-1]==shape_out[-1]
    //   quotients[-1]=0;
    //   quotients[-2]+=1;
    //   if quotients[-2]==shape_out[-2]
    //       quotients[-2]=0
    //       ...
    for (int32_t dim = shapeA_len - 1; dim >= 0; dim--) {
      if (quotients[dim] >= inputA[dim]) {
        quotients[dim] = 0;
        quotients[dim - 1] +=
            1; // should always stop before "quotients[0]==shape_out[0]" -->
               // "quotients[-1]"
      } else {
        break; // quotients are all correctly updated
      }
    }
    if (quotients[axis] ==
        (pSrcB[target] +
         offset)) { // sizeB==num_output if equal to 3. target does 0 1 2
      // printf("pREVUIOUSLY : %d\n", target);
      target = (target + 1) % num_output;
      // printf("Currently : %d\n\n", target);
      quotients[axis] = 0;
    }
    // printf("Targeting the %d output with pos %d\n and value %f\n\n",target,
    // pos[target], pSrcA[i]); if(target==1){
    //     printf("Pointer %p pos to write %d, value %f\n\n", pDstC[target],
    //     pos[target], pSrcA[i]);
    // }
    (pDstC[target])[pos[target]] = pSrcA[i];
    quotients[shapeA_len - 1] += 1;
    pos[target] = pos[target] + 1;
  }
}
