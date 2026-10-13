#include "DeeployBasicMath.h"
#include <math.h>

float32_t safe_get(const float32_t *__restrict__ pSrcA, int32_t N, int32_t C,
                   int32_t H, int32_t W, int32_t n, int32_t c, int32_t x,
                   int32_t y) {
  float32_t value = 0.0;
  if (x >= 0 && x < W && y >= 0 && y < H) {
    value = pSrcA[n * C * H * W + c * H * W + y * W + x];
  }
  return value;
}

void GridSample_fp32_fp32(const float32_t *__restrict__ pSrcA,
                          const float32_t *__restrict__ pSrcB,
                          float32_t *__restrict__ pDstC, int32_t *inputA,
                          int32_t shapeA_len, int32_t *inputB,
                          int32_t shapeB_len) {
  // all the ones we have are from v16
  // align corner : 0 (default)
  // mode bilinear (now linear) (still is the default)
  // padding mode zeros (default)

  // All the 4D NCHW case

  // shape_out
  int32_t shape_out[shapeA_len];
  int32_t mul_out = 1;
  // mul out doubles as size of output

  for (int32_t i = 0; i < shapeA_len; i++) {
    if (i == 0 || i == 1) {
      shape_out[i] = inputA[i];
    } else {
      shape_out[i] = inputB[i - 1];
    }
    mul_out = (mul_out)*shape_out[i];
  }

  int32_t posfinalA;
  int32_t posfinalB;

  // The fp values in B/grid are coordinates within to A/X indices
  // to then deduce the C/out value from all four corners values from A

  int32_t target;
  // All 4 corners means no way to efficiently traverse the A array

  float32_t u, d, l, r;     // up down left right
  float32_t ul, ur, dl, dr; // up-left up-right down-left down-right

  // quotients isn't useful?? C doesnt dictate our access to A, B does?
  // Let's traverse B instead
  int32_t N, C, H, W; // old
  N = inputA[0];
  C = inputA[1];
  H = inputA[2];
  W = inputA[3];

  for (int32_t n = 0; n < shape_out[0]; n++) {
    // for(int32_t c=0; c<shape_out[1];c++){
    for (int32_t h = 0; h < shape_out[2]; h++) {
      for (int32_t w = 0; w < shape_out[3]; w++) {
        posfinalA = 0;
        posfinalB = 0;

        target = n * shape_out[2] * shape_out[3] * 2 + h * shape_out[3] * 2 +
                 w * 2 + 0;
        // align corners tells us to do
        float32_t y = ((pSrcB[target + 1] + 1.0) * (float32_t)H - 1.0) / 2.0;
        float32_t x = ((pSrcB[target + 0] + 1.0) * (float32_t)W - 1.0) / 2.0;

        //"""
        // X ET Y SONT INVERSES X ET Y SONT INCVERSES X ET TY SONT INVERSES X
        // RET Y SONT INVERSES X ET Y SONT IN VERSES X ET UY SONT INVERSES!!!!!
        //"""

        // printf("\n\nthe converted coordinates : %f %f\n", y, x);
        // printf("they were : %f    %f \n\n",
        // pSrcB[n*shape_out[2]*shape_out[3]*2 + h*shape_out[3]*2 + w*2 + 1],
        // pSrcB[n*shape_out[2]*shape_out[3]*2 + h*shape_out[3]*2 + w*2 + 0]);

        int32_t ix = (int32_t)floor(x); // integer : the index in A
        int32_t iy = (int32_t)floor(y);

        u = y - iy; // up
        l = x - ix; // left
        // printf("Thus : %f %f\n", u, l);
        d = 1.0 - u; // down
        r = 1.0 - l; // right

        // mode (bi)linear tells us
        // recheck several times l.500
        // https://github.com/pytorch/pytorch/blob/9a0f7a3bb01b235ea04581ee540970a098071b72/aten/src/ATen/native/cpu/GridSamplerKernel.cpp#L497
        ul = d * r;
        ur = d * l;
        dl = u * r;
        dr = u * l;

        // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/test/contrib_ops/gridsample_test.cc
        // //

        // We are calculating the new values from
        // A[n][c][iy][ix]------A[n][c][iy][ix+1]
        //                                        A[n][c][iy+1][ix]----A[n][c][iy+1][ix+1]

        for (int32_t c = 0; c < shape_out[1];
             c++) { // can we optimize that loop ? all c at once?
          float32_t current;
          // int32_t offset;
          // offset=n*C*H*W + c*H*W + iy*W + ix;

          // https://github.com/OrkhanHI/pytorch_grid_sample_python/blob/main/pytorch_grid_sample_python.md
          // NEEEEED the safe get to handle padding automatically and FIX
          // EVERYTHING
          float32_t val0, val1, val2, val3;

          val0 = safe_get(pSrcA, N, C, H, W, n, c, ix, iy);
          val1 = safe_get(pSrcA, N, C, H, W, n, c, ix + 1, iy);
          val2 = safe_get(pSrcA, N, C, H, W, n, c, ix, iy + 1);
          val3 = safe_get(pSrcA, N, C, H, W, n, c, ix + 1, iy + 1);
          current = val0 * ul + val1 * ur + val2 * dl + val3 * dr;
          // printf("fetching in A at : %d with values %f \t%f\t %f\t %f \t"
          // ,offset, val0, val1, val2, val3);

          pDstC[n * shape_out[1] * shape_out[2] * shape_out[3] +
                c * shape_out[2] * shape_out[3] + h * shape_out[3] + w] =
              current;

          // printf("current value : %f\n", current);
        }
      }
    }
  }
  //}
}
