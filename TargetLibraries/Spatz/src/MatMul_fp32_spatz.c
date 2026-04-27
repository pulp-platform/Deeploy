#include "DeeploySpatzMath.h"
#include <snrt.h>

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

// void matmul(float *c, const float *a, const float *b, const unsigned int M,
//             const unsigned int N, const unsigned int P);

// void Spatz_MatMul_fp32_fp32_fp32(const float32_t *__restrict__ pSrcA,
//                                  const float32_t *__restrict__ pSrcB,
//                                  float32_t *__restrict__ pDstY, uint32_t M,
//                                  uint32_t N, uint32_t O) {
// 	// defined in ${SPATZ_HOME}/sw/spatzBenchmarks/sp-fmatmul/kernel/sp-fmatmul.c
//   matmul(pDstY, pSrcA, pSrcB, M, N, O);
// }

/*
a * b = c
*/
void Spatz_MatMul_fp32_fp32_fp32(const float32_t *__restrict__ a,
                                 const float32_t *__restrict__ b,
                                 float32_t *__restrict__ c, uint32_t M,
                                 uint32_t N, uint32_t P) {
  // const unsigned int num_cores = snrt_cluster_core_num(); = 2 for spatz
  const unsigned int cid = snrt_cluster_core_idx();

  unsigned int m_start = (M / 2) * cid;
  unsigned int m_end = (M / 2) * (cid + 1);

  if (M <= 4) {
    matmul_2xVL(c, a, b, m_start, m_end, N, P, 0, P);
  } else if (M <= 8) {
    matmul_4xVL(c, a, b, m_start, m_end, N, P, 0, P);
  } else {
    matmul_8xVL(c, a, b, m_start, m_end, N, P, 0, P);
  }
}
