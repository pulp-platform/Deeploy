#include "DeeployPULPMath.h"
#include "pmsis.h"


#define REDMULE_BASE_ADDR 0x10201C00 

#define REG_MNK_M         0x00
#define REG_MNK_N         0x04
#define REG_MNK_K         0x08
#define REG_X_ADDR        0x0C
#define REG_Y_ADDR        0x10
#define REG_Z_ADDR        0x14
#define REG_W_ADDR        0x18
#define REG_COMPUTE_MODE  0x1C
#define REG_TRIGGER       0x20
#define REG_WAIT          0x28

void MatMul_fp32_fp32_fp32_Redmule(
    const float32_t *__restrict__ pSrcA,
    const float32_t *__restrict__ pSrcB,
    float32_t *__restrict__ pDstY,
    uint32_t M,
    uint32_t N,
    uint32_t O) {
    
    uint32_t total_elements = M * O;
    for (uint32_t i = 0; i < total_elements; i++) {
        pDstY[i] = 0.0f;
    }
    
    volatile uint16_t *mnk_m = (volatile uint16_t *)(REDMULE_BASE_ADDR + REG_MNK_M);
    volatile uint16_t *mnk_n = (volatile uint16_t *)(REDMULE_BASE_ADDR + REG_MNK_N);
    volatile uint16_t *mnk_k = (volatile uint16_t *)(REDMULE_BASE_ADDR + REG_MNK_K);
    
    *mnk_m = (uint16_t)M;
    *mnk_n = (uint16_t)N;
    *mnk_k = (uint16_t)O;
    
    volatile uint32_t *x_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_X_ADDR);
    volatile uint32_t *y_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_Y_ADDR);
    volatile uint32_t *z_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_Z_ADDR);
    volatile uint32_t *w_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_W_ADDR);
    
    *x_addr = (uint32_t)((uintptr_t)pSrcA);
    *y_addr = (uint32_t)((uintptr_t)pDstY);
    *z_addr = (uint32_t)((uintptr_t)pDstY);
    *w_addr = (uint32_t)((uintptr_t)pSrcB);
    
    volatile uint32_t *compute_mode = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_COMPUTE_MODE);
    *compute_mode = 4;  // FP32 mode
    
    volatile uint32_t *trigger = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_TRIGGER);
    *trigger;  
    
    volatile uint32_t *wait_reg = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_WAIT);
    uint32_t result = *wait_reg;  
}



void Gemm_fp32_fp32_fp32_fp32_Redmule(
    const float32_t *__restrict__ pSrcA,
    const float32_t *__restrict__ pSrcB,
    const float32_t *__restrict__ pBias,
    float32_t *__restrict__ pDstY,
    uint32_t M,
    uint32_t N,
    uint32_t O) {

    
    volatile uint16_t *mnk_m = (volatile uint16_t *)(REDMULE_BASE_ADDR + REG_MNK_M);
    volatile uint16_t *mnk_n = (volatile uint16_t *)(REDMULE_BASE_ADDR + REG_MNK_N);
    volatile uint16_t *mnk_k = (volatile uint16_t *)(REDMULE_BASE_ADDR + REG_MNK_K);

    *mnk_m = (uint16_t)M;
    *mnk_n = (uint16_t)N;
    *mnk_k = (uint16_t)O;


    volatile uint32_t *x_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_X_ADDR);
    volatile uint32_t *y_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_Y_ADDR);
    volatile uint32_t *z_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_Z_ADDR);
    volatile uint32_t *w_addr = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_W_ADDR);

    *x_addr = (uint32_t)((uintptr_t)pSrcA);
    *y_addr = (uint32_t)((uintptr_t)pBias);  
    *z_addr = (uint32_t)((uintptr_t)pDstY);
    *w_addr = (uint32_t)((uintptr_t)pSrcB);

    volatile uint32_t *compute_mode = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_COMPUTE_MODE);
    *compute_mode = 4;  // FP32 mode
    
    volatile uint32_t *trigger = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_TRIGGER);
    *trigger;  
    
    volatile uint32_t *wait_reg = (volatile uint32_t *)(REDMULE_BASE_ADDR + REG_WAIT);
    uint32_t result = *wait_reg;  
}

void Conv2d_Im2Col_fp32_fp32_fp32_HWC_8_Redmule(
    const float32_t *__restrict__ pSrcA,
    uint32_t H,
    uint32_t W,
    uint32_t C,
    const float32_t *__restrict__ pSrcB,
    uint32_t P,
    uint32_t Q,
    uint32_t SP,
    uint32_t SQ,
    float32_t *__restrict__ pDstC,
    uint32_t F,
    uint32_t pad_top,
    uint32_t pad_bottom,
    uint32_t pad_left,
    uint32_t pad_right,
    float32_t *__restrict__ pIm2ColBuffer) {
    
    uint32_t H_out = (H + pad_top + pad_bottom - P) / SP + 1;
    uint32_t W_out = (W + pad_left + pad_right - Q) / SQ + 1;
    uint32_t kernel_size = P * Q * C;
    uint32_t core_id = pi_core_id();
    uint32_t num_cores = NUM_CORES;
    
    uint32_t total_positions = H_out * W_out;
    uint32_t num_batches = (total_positions + num_cores - 1) / num_cores;

    float32_t *core_im2col_buffer = pIm2ColBuffer + core_id * kernel_size;
    
    for (uint32_t batch = 0; batch < num_batches; batch++) {

        uint32_t batch_start_pos = batch * num_cores;
        

        uint32_t valid_cores = MIN(num_cores, total_positions - batch_start_pos);
        

        if (core_id < valid_cores) {

            uint32_t pos = batch_start_pos + core_id;
            

            uint32_t h_out = pos / W_out;
            uint32_t w_out = pos % W_out;
            int32_t h_in_start = h_out * SP - pad_top;
            int32_t w_in_start = w_out * SQ - pad_left;
            

            for (uint32_t p = 0; p < P; p++) {
                int32_t h_in = h_in_start + p;
                
                for (uint32_t q = 0; q < Q; q++) {
                    int32_t w_in = w_in_start + q;
                    uint32_t in_offset = (h_in * W + w_in) * C;
                    uint32_t kernel_offset = (p * Q + q) * C;
                    
                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        
                        for (uint32_t c = 0; c < C; c++) {
                            core_im2col_buffer[kernel_offset + c] = pSrcA[in_offset + c];
                        }
                    }
                    else {
                        
                        for (uint32_t c = 0; c < C; c++) {
                            core_im2col_buffer[kernel_offset + c] = 0.0f;
                        }
                    }

                }
            }
        }
        

        pi_cl_team_barrier();
        

        if (core_id == 0) {

            float32_t *batch_output = pDstC + batch_start_pos * F;

            MatMul_fp32_fp32_fp32_Redmule(
                pIm2ColBuffer,     
                pSrcB,             
                batch_output,      
                valid_cores,       
                kernel_size,       
                F                  
            );
        }
        
        pi_cl_team_barrier();
    }
}