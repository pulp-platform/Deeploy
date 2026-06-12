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



void gemv_v32b_m4(float *a, float *b, float *c, int N, int P) {
  unsigned int p = 0;
  while (p < (unsigned int)P) {
    size_t gvl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                 : "=r"(gvl) : "r"((unsigned int)P - p));

    const float *b_ = b + p;                       
    asm volatile("vle32.v v8, (%0)" ::"r"(b_));
    asm volatile("vfmul.vf v16, v8, %0" ::"f"(a[0]));   

    for (int row = 1; row < N; row++) {
      b_ += P;                                     
      asm volatile("vle32.v v8, (%0)" ::"r"(b_));
      asm volatile("vfmacc.vf v16, %0, v8" ::"f"(a[row]));  
    }

    asm volatile("vse32.v v16, (%0)" ::"r"(c + p));
    p += gvl;
  }
}


void Spatz_MatMul_fp32_fp32_fp32(const float32_t *__restrict__ a,
                                 const float32_t *__restrict__ b,
                                 float32_t *__restrict__ c, uint32_t M,
                                 uint32_t N, uint32_t P) {
  // const unsigned int num_cores = snrt_cluster_core_num(); = 2 for spatz
  const unsigned int cid = snrt_cluster_core_idx();

  if (M == 1) {
    printf("a: 0x%x, b: 0x%x, c: 0x%x\n", (uint32_t)a, (uint32_t)b, (uint32_t)c);
      if (cid==0){
       gemv_v32b_m4(a, b, c, N, P);
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
