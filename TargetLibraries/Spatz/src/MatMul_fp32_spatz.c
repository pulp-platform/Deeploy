#include "DeeploySpatzMath.h"
#include <snrt.h>
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

void gemv_v32b_m4_dual_core(float *a, float *b, float *c, int N, int local_P, int total_P) {
  unsigned int p = 0;
  // Loop only up to the local number of columns assigned to this core
  while (p < (unsigned int)local_P) {
    size_t gvl;
    asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                 : "=r"(gvl) : "r"((unsigned int)local_P - p));

    const float *b_ = b + p;                        
    asm volatile("vle32.v v8, (%0)" ::"r"(b_));
    asm volatile("vfmul.vf v16, v8, %0" ::"f"(a[0]));   

    for (int row = 1; row < N; row++) {
      // CRITICAL: Must skip the TOTAL width of the matrix to reach the next row
      b_ += total_P;
      asm volatile("vle32.v v8, (%0)" ::"r"(b_));
      asm volatile("vfmacc.vf v16, %0, v8" ::"f"(a[row]));
    }

    asm volatile("vse32.v v16, (%0)" ::"r"(c + p));
    p += gvl;
  }
}

void gemv_col_reduction(float *a, float *b, float *c, int N, int P) {
  // Stride between elements in the same column of matrix B (in bytes)
  ptrdiff_t b_stride = P * sizeof(float);

  // Process one column of the matrix (and one element of result C) at a time
  for (int col = 0; col < P; col++) {
    unsigned int row = 0;
    
    // Clear vector register v0 (takes v0-v7) to accumulate partial products
    size_t init_gvl;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(init_gvl));
    asm volatile("vmv.v.i v0, 0"); 

    // Loop through the N elements of the current column
    while (row < (unsigned int)N) {
      size_t gvl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(gvl) : "r"((unsigned int)N - row));

      // Pointer to the current piece of the column in B and vector A
      const float *b_ptr = b + (row * P) + col;
      const float *a_ptr = a + row;

      // Load strided elements from B into v8 (takes v8-v15)
      asm volatile("vlse32.v v8, (%0), %1" ::"r"(b_ptr), "r"(b_stride));
      
      // Load contiguous elements from A into v16 (takes v16-v23)
      asm volatile("vle32.v v16, (%0)" ::"r"(a_ptr));

      // Multiply and accumulate: v0 = v0 + (v8 * v16)
      asm volatile("vfmacc.vv v0, v8, v16");

      row += gvl;
    }

    // --- Reduction Phase ---
    // For reductions, the scalar destination register group typically must be m1 (or matching).
    // To be perfectly safe, we set the vector length to 1 element for the scalar target v24.
    asm volatile("vsetvli zero, zero, e32, m8, ta, ma");
    asm volatile("vmv.v.i v24, 0");

    // Reduce the accumulated vector v0 into the first element of v24
    asm volatile("vfredosum.vs v24, v0, v24");

    // Store only the single scalar result (1 element) into c[col]
    // We explicitly set vl=1 to avoid overwriting memory past c[col] due to m8 group sizing
    size_t one = 1;
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"(one));
    asm volatile("vse32.v v24, (%0)" ::"r"(c + col));
  }
}

void gemv_col_reduction_dual_core(float *a, float *b, float *c, int N, int local_P, int total_P) {
  // CRITICAL: Stride must use the original TOTAL width of matrix B
  ptrdiff_t b_stride = total_P * sizeof(float);

  // Loop only through the columns assigned to this core
  for (int col = 0; col < local_P; col++) {
    unsigned int row = 0;
    
    // Clear vector register v0 (takes v0-v7) to accumulate partial products
    size_t init_gvl;
    asm volatile("vsetvli %0, zero, e32, m8, ta, ma" : "=r"(init_gvl));
    asm volatile("vmv.v.i v0, 0"); 

    // Loop through the N elements of the current column
    while (row < (unsigned int)N) {
      size_t gvl;
      asm volatile("vsetvli %0, %1, e32, m8, ta, ma"
                   : "=r"(gvl) : "r"((unsigned int)N - row));

      // Pointer uses total_P to correctly jump down the rows
      const float *b_ptr = b + (row * total_P) + col;
      const float *a_ptr = a + row;

      // Load strided elements from B into v8 (takes v8-v15)
      asm volatile("vlse32.v v8, (%0), %1" ::"r"(b_ptr), "r"(b_stride));
      
      // Load contiguous elements from A into v16 (takes v16-v23)
      asm volatile("vle32.v v16, (%0)" ::"r"(a_ptr));

      // Multiply and accumulate: v0 = v0 + (v8 * v16)
      asm volatile("vfmacc.vv v0, v8, v16");

      row += gvl;
    }

    // --- Reduction Phase ---
    asm volatile("vsetvli zero, zero, e32, m8, ta, ma");
    asm volatile("vmv.v.i v24, 0");

    // Reduce the accumulated vector v0 into the first element of v24
    asm volatile("vfredosum.vs v24, v0, v24");

    // Store only the single scalar result (1 element) into c[col]
    size_t one = 1;
    asm volatile("vsetvli zero, %0, e32, m1, ta, ma" :: "r"(one));
    asm volatile("vse32.v v24, (%0)" ::"r"(c + col));
  }
}

void matmul_naive_vanilla(float *a, float *b, float *c, int M, int N, int P) {
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < P; j++) {
      float sum = 0.0f;
      for (int k = 0; k < N; k++) {
        sum += a[i * N + k] * b[k * P + j];
      }
      
      c[i * P + j] = sum;
    }
  }
}



void Spatz_MatMul_fp32_fp32_fp32(const float32_t *__restrict__ a,
                                 const float32_t *__restrict__ b,
                                 float32_t *__restrict__ c, uint32_t M,
                                 uint32_t N, uint32_t P) {
  // const unsigned int num_cores = snrt_cluster_core_num(); = 2 for spatz
  const unsigned int cid = snrt_cluster_core_idx();

  if (M == 1) {
    // TODO make this be more specific, probably needs to me N>5*P or some other constant
    int cols_core0 = P / 2;
    int cols_core1 = P - cols_core0; // Safely gets the remainder if P is odd
    if (N>4*P){
      if (cid == 0) {
          gemv_col_reduction_dual_core(a, b, c, N, cols_core0, P);
      } else {
          float *b_offset = b + cols_core0; float *c_offset = c + cols_core0;
          gemv_col_reduction_dual_core(a, b_offset, c_offset, N, cols_core1, P);
      }
      // if (cid == 0) { gemv_col_reduction(a, b, c, N, P); }
    } else {
      if (cid == 0) {
        gemv_v32b_m4_dual_core(a, b, c, N, cols_core0, P);
      } else {
        float *b_offset = b + cols_core0; float *c_offset = c + cols_core0;
        gemv_v32b_m4_dual_core(a, b_offset, c_offset, N, cols_core1, P);
      }
      // if (cid == 0) { gemv_v32b_m4(a, b, c, N, P); } 
    }
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
