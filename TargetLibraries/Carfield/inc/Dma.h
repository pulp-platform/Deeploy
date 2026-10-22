// system DMA

#ifndef __CARFIELD_DMA_H_
#define __CARFIELD_DMA_H_

#include <stdint.h>
#include "pulp.h"

// __base_dma = 0x01000000 (cheshire/sw/link/common.ldh).
// Offset from cheshire/sw/include/regs/idma.h.
// 64 bit register on a 32 bit bus: lower word at offset, higher word at offset+4.
// Cluster addresses are always 32 bit so the higher word is always 0.

// While running, reading this will launch the transfer
#define DMA_BASE      0x01000000U
#define DMA_CONF      (DMA_BASE + 0x00)
#define DMA_STATUS    (DMA_BASE + 0x04)
#define DMA_NEXT_ID   (DMA_BASE + 0x44)
#define DMA_DONE      (DMA_BASE + 0x84)
#define DMA_DST_LO    (DMA_BASE + 0xD0)
#define DMA_SRC_LO    (DMA_BASE + 0xD8)
#define DMA_LEN_LO    (DMA_BASE + 0xE0)
#define DMA_DSTSTR_LO (DMA_BASE + 0xE8)
#define DMA_SRCSTR_LO (DMA_BASE + 0xF0)
#define DMA_REPS_LO   (DMA_BASE + 0xF8)

#define DMA_CONF_1D   0x0U
#define DMA_CONF_2D   (1U << 10)

static inline uint32_t dma_1d(uint32_t dst, uint32_t src, uint32_t nbytes) {
    pulp_write32(DMA_SRC_LO, src);    pulp_write32(DMA_SRC_LO + 4, 0); // write lower part of dma reg, and all zeros to higher part
    pulp_write32(DMA_DST_LO, dst);    pulp_write32(DMA_DST_LO + 4, 0);
    pulp_write32(DMA_LEN_LO, nbytes); pulp_write32(DMA_LEN_LO + 4, 0);
    pulp_write32(DMA_CONF,   DMA_CONF_1D);
    __asm__ volatile("" ::: "memory"); // barrier
    return pulp_read32(DMA_NEXT_ID); // transfer start after reading
}

// No stride in L1 to avoid wasting space, but the stride is conserved.
static inline uint32_t dma_2d(uint32_t dst, uint32_t src, uint32_t line,
                              uint32_t src_stride, uint32_t dst_stride,
                              uint32_t reps) {
    pulp_write32(DMA_SRC_LO,    src);        pulp_write32(DMA_SRC_LO + 4, 0);
    pulp_write32(DMA_DST_LO,    dst);        pulp_write32(DMA_DST_LO + 4, 0);
    pulp_write32(DMA_LEN_LO,    line);       pulp_write32(DMA_LEN_LO + 4, 0);
    pulp_write32(DMA_CONF,      DMA_CONF_2D);
    pulp_write32(DMA_SRCSTR_LO, src_stride); pulp_write32(DMA_SRCSTR_LO + 4, 0);
    pulp_write32(DMA_DSTSTR_LO, dst_stride); pulp_write32(DMA_DSTSTR_LO + 4, 0);
    pulp_write32(DMA_REPS_LO,   reps);       pulp_write32(DMA_REPS_LO + 4, 0);
    __asm__ volatile("" ::: "memory");
    return pulp_read32(DMA_NEXT_ID);
}

// DONE is a counter, not a flag
static inline void dma_wait(uint32_t id) {
    while ((int32_t)(pulp_read32(DMA_DONE) - id) < 0);
}

// Guard: it will give 0 if transferring does not complete. Without it the cluster
// would hang because of a wrong ID or a blocked DMA.
static inline int dma_wait_guarded(uint32_t id, uint32_t max_iter) {
    while ((int32_t)(pulp_read32(DMA_DONE) - id) < 0 && --max_iter);
    return max_iter != 0;
}

#endif // __CARFIELD_DMA_H_
