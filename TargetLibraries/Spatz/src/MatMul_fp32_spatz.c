#include "DeeploySpatzMath.h"
#include <snrt.h>
#include "printf.h"
#include <stdlib.h>

// functions defined in ${SPATZ_HOME}/sw/spatzBenchmarks/sp-fmatmul/kernel/sp-fmatmul.c
void matmul_2xVL(float *c, const float *a, const float *b,
                 const unsigned int m_start, const unsigned int m_end,
                 const unsigned int N, const unsigned int P,
                 const unsigned int p_start, const unsigned int p_end);


void matmul_4xVL(float *c, const float *a, const float *b,
                 const unsigned int m_start, const unsigned int m_end,
                 const unsigned int N, const unsigned int P,
                 const unsigned int p_start, const unsigned int p_end);


void matmul_8xVL(float *c, const float *a, const float *b,
                 const unsigned int m_start, const unsigned int m_end,
                 const unsigned int N, const unsigned int P,
                 const unsigned int p_start, const unsigned int p_end);

/*
This function calculates:
if a is stored in row major format
 b * a = c
1xm mxn 1xn
*/
void gemv_v32b_m4(float *a, float *b, float *c, int M, int N_core, int N) {
  unsigned int vl, avl = N_core;
  float *a_, *a_start = a;
  float *b_ = b;
  float *c_ = c;

  do {
    a_ = a_start;
    asm volatile("vsetvli %0, %1, e32, m4, ta, ma" : "=r"(vl) : "r"(avl));
    
    printf("vl=%d avl=%d\n", vl, avl);
    for (int row = 0; row < M; row += 1) {
      // printf("processing row %d and the next\n", row);
      // printf("*a_=%08x,*b_=%08x\n", *(uint32_t*)a_, *(uint32_t*)b_);
      // Load chunk a
      asm volatile("vle32.v v8, (%0)" ::"r"(a_));

      // Multiply and accumulate
      if (row == 0) { asm volatile("vfmul.vf v4, v8, %0" ::"f"(*b_));
      } else { asm volatile("vfmacc.vf v4, %0, v8" ::"f"(*b_)); }
      a_ += N;
      b_++;

      // Load chunk a
      // printf("*a_=%08x,*b_=%08x\n", *(uint32_t*)a_, *(uint32_t*)b_);
//       asm volatile("vle32.v v8, (%0)" ::"r"(a_));
// 
//       // Multiply and accumulate
//       if (row == 0) { asm volatile("vfmul.vf v12, v8, %0" ::"f"(*b_));
//       } else { asm volatile("vfmacc.vf v12, %0, v8" ::"f"(*b_)); }
//       a_ += N;
//       b_++;
    }
    // asm volatile("vfadd.vv v12, v12, v4");
    asm volatile("vse32.v v4, (%0)" ::"r"(c_));
    // if (snrt_is_dm_core()){printf("*c_=%08x", *(uint32_t*)c_);}
    // printf("c[x..x+vl]=");
    // for (int i=0;i<vl;i++){ printf("%08x\n", *(uint32_t*)(c_+i)); }
    // printf("\n");
    avl -= vl;
    c_ += vl;
    b_ = b;
    a_start += vl;
  } while (avl > 0);
}

float fdotp_v32b(const float *a, const float *b, unsigned int avl) {
  const unsigned int orig_avl = avl;
  unsigned int vl;

  float red;

  asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
  asm volatile("vmv.s.x v0, zero");

  // Stripmine and accumulate a partial reduced vector
  do {
    // Set the vl
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma" : "=r"(vl) : "r"(avl));
    printf("avl=%d, vl=%d\n", avl, vl);

    // Load chunk a and b
    asm volatile("vle32.v v8,  (%0)" ::"r"(a));
    asm volatile("vle32.v v16, (%0)" ::"r"(b));

    // Multiply and accumulate
    if (avl == orig_avl) {
      asm volatile("vfmul.vv v24, v8, v16");
    } else {
      asm volatile("vfmacc.vv v24, v8, v16");
    }

    // Bump pointers
    a += vl;
    b += vl;
    avl -= vl;
  } while (avl > 0);

  // Reduce and return
  // Debug: dump contents of v24 before reduction


  asm volatile("vsetvli zero, %0, e32, m8, ta, ma" ::"r"(orig_avl));
  asm volatile("vfredusum.vs v0, v24, v0");
  asm volatile("vfmv.f.s %0, v0" : "=f"(red));
  printf("c=%08x\n",red);
  return red;
}

void Spatz_MatMul_fp32_fp32_fp32(const float32_t *__restrict__ a,
                                 const float32_t *__restrict__ b,
                                 float32_t *__restrict__ c, uint32_t M,
                                 uint32_t N, uint32_t P) {
  // const unsigned int num_cores = snrt_cluster_core_num(); = 2 for spatz
  const unsigned int cid = snrt_cluster_core_idx();

  if (M == 1) {
    // if (P==1){
    //   if (cid==0){
    //     float32_t res = fdotp_v32b(a, b, N);
    //   }
    // } else {
      // here b is the matrix, a is the vector, opposite to the names in the function gemv_v32b_m4
      // here: M=1, N, P; in gemv_v32b_m4: 1 implied, M, N; so N_here = M_func, P_here = N_func
      // const unsigned int columns_per_core = P/2;
      // float32_t *matrix_core = b + columns_per_core * cid;
      // float32_t *result_core = c + cid * columns_per_core;
      // gemv_v32b_m4(matrix_core, a, result_core, N, columns_per_core, P);
      if (cid==0){
        gemv_v32b_m4(b, a, c, N, P, P); 
        printf("c is:\n");
        for (int i=0;i<P;i++){ printf("%08x\n", *(uint32_t*)(c+i)); }
      }
    // }
  } else {
    unsigned int p_start, p_end;
    if (cid == 0){ p_start = 0; p_end = (P/2);
    } else { p_start = (P/2); p_end = P; }

    if (M <= 4) {
      matmul_2xVL(c, a, b, 0, M, N, P, p_start, p_end);
    } else if (M <= 8) {
      matmul_4xVL(c, a, b, 0, M, N, P, p_start, p_end);
    } else {
      matmul_8xVL(c, a, b, 0, M, N, P, p_start, p_end);
    }
  }
}
