
// Tiled FP32 2D convolution for the Carfield PULP cluster: it stores tiles in L1 of cluster directly (not passing through L2).
// The naive reference kernel is in Conv.c, which keeps all tensors in DDR.

// out_tile is written back only when the sum over all C channels is complete.

// (NF, R, Wt, Cb) chosen to store a tile in L1, leaving about 12KiB of headroom.

#include <stdio.h>

#include "Conv2D_tiled.h"
#include "Utils.h"
#include "Dma.h"

#define TILE_NF 32 // filters per tile
#define TILE_R   8 // output tile rows
#define TILE_WT 80 // output tile columns
#define TILE_CB  4 // input channels per block

// Memory alignment since DDR and LLC are aligned to 64 too. pi_l1_malloc only
// guarantees 8, so we over-allocate and round up by hand.
#define ALIGN64(tileCol)  (((tileCol) + 63u) & ~63u)

#define MINI(a, b)  ((a) < (b) ? (a) : (b))

// Guard to check DMA failure
#define DMA_GUARD_ITER 10000000U

// uint8 for offset arithmetic.
static uint8_t *tileRaw = NULL;
static uint8_t *tileL1 = NULL;

// If the transfer fails, print the reason and signal the EOC to the host.
static void tiledDie(const char *why, uint32_t value) {
    printf("Carfield_Conv2D_tiled: %s (%unit)\n\r", why, (unsigned)value);
    hal_cluster_ctrl_return_set_remote(hal_cluster_id(), 0x7FFFFFFF);
    hal_cluster_ctrl_eoc_set_remote(hal_cluster_id(), 1);
    while (1);
}

void Carfield_Conv2D_tiled(float32_t *input, float32_t *weight, float32_t *bias,
    float32_t *output, int32_t F, int32_t C, int32_t H_in, int32_t W_in,
    int32_t P, int32_t Q, int32_t core_id, int32_t numThreads) {

    // The input arrives already padded, exactly as Carfield_Conv expects it.
    const int32_t H_out = H_in - P + 1; // output tensor rows
    const int32_t W_out = W_in - Q + 1; // output tensor columns

    // Higher tile allocation within the call to the kernel.
    const int32_t maxNF = MINI(TILE_NF, F);
    const int32_t maxR = MINI(TILE_R, H_out);
    const int32_t maxWt = MINI(TILE_WT, W_out);
    const int32_t maxCb = MINI(TILE_CB, C);

    // Bytes in each of the four L1 buffers, sized on the largest tile.
    const uint32_t weightBytes = (uint32_t)(maxNF * maxCb * P * Q) * 4u;
    const uint32_t biasBytes = (uint32_t)F * 4u;
    const uint32_t inputBytes = (uint32_t)(maxCb * (maxR + P - 1) * (maxWt + Q - 1)) * 4u;
    const uint32_t outputBytes = (uint32_t)(maxNF * maxR * maxWt) * 4u;

    // Every offset is the previous offset plus the size of the buffer that
    // lives there, and rounded up to 64-byte alignment.
    const uint32_t weightOff = 0u;
    const uint32_t biasOff = weightOff + ALIGN64(weightBytes);
    const uint32_t inputOff = biasOff + ALIGN64(biasBytes);
    const uint32_t outputOff = inputOff + ALIGN64(inputBytes);
    const uint32_t totBytes = outputOff + ALIGN64(outputBytes);

    // Allocation + bias import.
    BEGIN_SINGLE_CORE
    tileRaw = (uint8_t *)pi_l1_malloc(0, totBytes + 63u); // Over-allocate to align by hand.

    // Two issues, two guards.
    if (tileRaw == NULL) {
        tiledDie("pi_l1_malloc failed for bytes", totBytes + 63u);
    }

    tileL1 = (uint8_t *)ALIGN64((uint32_t)tileRaw); // Aligning by hand.

    if ((uint32_t)tileL1 + totBytes > CARFIELD_L1_END) {
        tiledDie("L1 tile overruns usable heap by bytes",
                 (uint32_t)tileL1 + totBytes - CARFIELD_L1_END);
    }

    // The bias vector is not tiled.
    uint32_t idBias = dma_1d((uint32_t)(tileL1 + biasOff), (uint32_t)bias, biasBytes);
    if (!dma_wait_guarded(idBias, DMA_GUARD_ITER)) {
        tiledDie("DMA timeout on bias", 0u);
    }
    END_SINGLE_CORE

    synch_barrier();

    // Pointers to every section in L1.
    float32_t *const weightL1 = (float32_t *)(tileL1 + weightOff);
    float32_t *const biasL1 = (float32_t *)(tileL1 + biasOff);
    float32_t *const inputL1 = (float32_t *)(tileL1 + inputOff);
    float32_t *const outputL1 = (float32_t *)(tileL1 + outputOff);

    // Cycling on every tile until the end of the tensor, and the last tile shrinks to fit.
    // Legend: (nf, r, wt, cb) = (filters, rows, columns, channels) in the current tile.
    // f0 = starting point, F = end of tensor.
    for (int32_t f0 = 0; f0 < F; f0 += TILE_NF) {
        const int32_t nf = MINI(TILE_NF, F - f0);

        for (int32_t h0 = 0; h0 < H_out; h0 += TILE_R) {
            const int32_t r = MINI(TILE_R, H_out - h0);
            const int32_t ih = r + P - 1;  // input rows current tile needs

            for (int32_t w0 = 0; w0 < W_out; w0 += TILE_WT) {
                const int32_t wt = MINI(TILE_WT, W_out - w0);
                const int32_t iw = wt + Q - 1;  // input columns current tile needs

                for (int32_t c0 = 0; c0 < C; c0 += TILE_CB) {
                    const int32_t cb = MINI(TILE_CB, C - c0);

                    BEGIN_SINGLE_CORE
                    // Input: dma_2d carries one source stride, and in NCHW the
                    // jump between rows (W_in) differs from the jump between
                    // channels (H_in*W_in), so in general it takes one command
                    // per channel. When the tile spans the whole width (iw==W_in),
                    // the rows of a channel are contiguous and all cb channels
                    // collapse into a single command.
                    if (iw == W_in) {
                        dma_2d((uint32_t)inputL1,                             // dst
                               (uint32_t)(input + (c0 * H_in + h0) * W_in),   // src
                               (uint32_t)(ih * W_in) * 4u,                    // line = ih entire rows of a channel transferred per command
                               (uint32_t)(H_in * W_in) * 4u,                  // src_stride = a channel
                               (uint32_t)(ih * W_in) * 4u,                    // dst_stride = line
                               (uint32_t)cb);                                 // reps = cb channels
                    } else {
                        for (int32_t c = 0; c < cb; c++) {
                            dma_2d((uint32_t)(inputL1 + c * ih * iw),                       // dst
                                   (uint32_t)(input + ((c0 + c) * H_in + h0) * W_in + w0),  // src
                                   (uint32_t)iw * 4u,                                       // line = iw elements of a row
                                   (uint32_t)W_in * 4u,                                     // src_stride = a row
                                   (uint32_t)iw * 4u,                                       // dst_stride = line
                                   (uint32_t)ih);                                           // reps = ih rows of a channel
                        }
                    }

                    // Weights: in [F][C][P][Q] the cb channels of one filter are
                    // contiguous and consecutive filters are C*P*Q apart, so the
                    // whole tile of weights is always a single 2D command.
                    uint32_t idW = dma_2d((uint32_t)weightL1,                            // dst
                                          (uint32_t)(weight + (f0 * C + c0) * P * Q),    // src
                                          (uint32_t)(cb * P * Q) * 4u,                   // line = cb*P*Q elements of a filter
                                          (uint32_t)(C * P * Q) * 4u,                    // src_stride = a filter
                                          (uint32_t)(cb * P * Q) * 4u,                   // dst_stride = line
                                          (uint32_t)nf);                                 // reps = nf filters

                    if (!dma_wait_guarded(idW, DMA_GUARD_ITER)) {
                        tiledDie("DMA timeout staging a tile", 0u);
                    }
                    END_SINGLE_CORE

                    synch_barrier();

                    // The output tile holds nf*r rows of wt elements. Cores take
                    // those rows flattened: unit = filter*r + row. Every core takes
                    // rows.
                    // Rows are disjoint in L1, so no races.
                    const int32_t units = nf * r;
                    for (int32_t unit = core_id; unit < units; unit += numThreads) {
                        const int32_t tileFilter = unit / r;              // which filter in the tile
                        const int32_t tileRow = unit - tileFilter * r;    // which row in the tile

                        float32_t *const outputRow = outputL1 + unit * wt;                         // pointer to the row in the output tile
                        const float32_t *const weightFilter = weightL1 + tileFilter * cb * P * Q;  // weights of filter tileFilter in the tile

                        for (int32_t tileCol = 0; tileCol < wt; tileCol++) {
                            // First channel block seeds the accumulator with the
                            // bias, the others add to what is already there.
                            float32_t acc = (c0 == 0) ? biasL1[f0 + tileFilter] : outputRow[tileCol];

                            for (int32_t c = 0; c < cb; c++) {
                                const float32_t *const inputChannel = inputL1 + c * ih * iw;       // window of channel c
                                const float32_t *const weightChannel = weightFilter + c * P * Q;   // 3x3 of channel c in filter tileFilter
                                // window
                                for (int32_t p = 0; p < P; p++) {
                                    const float32_t *const inputRow = inputChannel + (tileRow + p) * iw + tileCol;
                                    const float32_t *const weightRow = weightChannel + p * Q;
                                    for (int32_t q = 0; q < Q; q++) {
                                        acc += inputRow[q] * weightRow[q];
                                    }
                                }
                            }
                            outputRow[tileCol] = acc;
                        }
                    }

                    synch_barrier();
                }

                // The sum over all C channels is complete and the tile can leave.
                BEGIN_SINGLE_CORE
                uint32_t idOut = 0u;
                if (wt == W_out) {
                    // Full-width tile: the r rows of a filter are contiguous in
                    // DDR, so all nf filters are one command.
                    idOut = dma_2d((uint32_t)(output + (f0 * H_out + h0) * W_out),
                                   (uint32_t)outputL1,
                                   (uint32_t)(r * W_out) * 4u,
                                   (uint32_t)(r * W_out) * 4u,
                                   (uint32_t)(H_out * W_out) * 4u,
                                   (uint32_t)nf);
                } else {
                    for (int32_t f = 0; f < nf; f++) {
                        idOut = dma_2d((uint32_t)(output + ((f0 + f) * H_out + h0) * W_out + w0),
                                       (uint32_t)(outputL1 + f * r * wt),
                                       (uint32_t)wt * 4u,
                                       (uint32_t)wt * 4u,
                                       (uint32_t)W_out * 4u,
                                       (uint32_t)r);
                    }
                }
                if (!dma_wait_guarded(idOut, DMA_GUARD_ITER)) {
                    tiledDie("DMA timeout writing a tile", 0u);
                }
                END_SINGLE_CORE

                synch_barrier();
            }
        }
    }
    // Freeing the raw pointer allocated by pi_l1_malloc, not the aligned one.
    SINGLE_CORE pi_l1_free(0, tileRaw, totBytes + 63u);
}
