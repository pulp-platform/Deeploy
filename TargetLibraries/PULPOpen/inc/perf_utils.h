/*
 * Performance Counter Utilities for PULP Benchmarking
 */

#ifndef __PERF_UTILS_H__
#define __PERF_UTILS_H__

#include "pmsis.h"

#define CONFIG_GVSOC_ISS_EXTERNAL_PCCR 1
#define CSR_PCER_CYCLES   0  /* Count the number of cycles the core was running */
#define CSR_PCER_INSTR    1  /* Count the number of instructions executed */
#define CSR_PCER_LD_STALL   2  /* Number of load use hazards */
#define CSR_PCER_JMP_STALL    3  /* Number of jump register hazards */
#define CSR_PCER_IMISS    4  /* Cycles waiting for instruction fetches. i.e. the number of instructions wasted due to non-ideal caches */
#define CSR_PCER_LD   5  /* Number of memory loads executed. Misaligned accesses are counted twice */
#define CSR_PCER_ST   6  /* Number of memory stores executed. Misaligned accesses are counted twice */
#define CSR_PCER_JUMP   7  /* Number of jump instructions seen, i.e. j, jr, jal, jalr */
#define CSR_PCER_BRANCH   8  /* Number of branch instructions seen, i.e. bf, bnf */
#define CSR_PCER_TAKEN_BRANCH 9  /* Number of taken branch instructions seen, i.e. bf, bnf */
#define CSR_PCER_RVC    10  /* Number of compressed instructions */
#define CSR_PCER_ELW    11  /* Cycles wasted due to ELW instruction */

#if defined(CONFIG_GVSOC_ISS_EXTERNAL_PCCR)
#define CSR_PCER_LD_EXT   12  /* Number of memory loads to EXT executed. Misaligned accesses are counted twice. Every non-TCDM access is considered external */
#define CSR_PCER_ST_EXT   13  /* Number of memory stores to EXT executed. Misaligned accesses are counted twice. Every non-TCDM access is considered external */
#define CSR_PCER_LD_EXT_CYC 14  /* Cycles used for memory loads to EXT. Every non-TCDM access is considered external */
#define CSR_PCER_ST_EXT_CYC 15  /* Cycles used for memory stores to EXT. Every non-TCDM access is considered external */
#define CSR_PCER_TCDM_CONT  16  /* Cycles wasted due to TCDM/log-interconnect contention */
#define CSR_PCER_APU_TY_CONF 17
#define CSR_PCER_APU_CONT    18
#define CSR_PCER_APU_DEP     19
#define CSR_PCER_APU_WB      20
#else
#define CSR_PCER_APU_TY_CONF 13
#define CSR_PCER_APU_CONT    14
#define CSR_PCER_APU_DEP     15
#define CSR_PCER_APU_WB      16
#endif

#define CSR_PCER_NB_EVENTS               21
#if defined(CONFIG_GVSOC_ISS_EXTERNAL_PCCR)
#define CSR_PCER_FIRST_EXTERNAL_EVENTS   12
#define CSR_PCER_NB_EXTERNAL_EVENTS      5
#define CSR_PCER_FIRST_APU_EVENTS        17
#define CSR_PCER_NB_APU_EVENTS           4
#else
#define CSR_PCER_FIRST_APU_EVENTS        13
#define CSR_PCER_NB_APU_EVENTS           4
#endif
#define CSR_NB_PCCR             31

// Gives from the event ID, the HW mask that can be stored (with an OR with other events mask) to the PCER
#define CSR_PCER_EVENT_MASK(eventId)  (1<<(eventId))
#define CSR_PCER_ALL_EVENTS_MASK  0xffffffff

// #define CSR_PCMR_ACTIVE           0x1 /* Activate counting */
// #define CSR_PCMR_SATURATE         0x2 /* Activate saturation */

#define CSR_PCER_NAME(id) (id == 0 ? "Cycles" : id == 1 ? "Instructions" : id == 2 ? "LD_Stall" : id == 3 ? "Jmp_Stall" : id == 4 ? "IMISS" : id == 5 ? "LD" : id == 6 ? "ST" : id == 7 ? "JUMP" : id == 8 ? "BRANCH" : id == 9 ? "TAKEN_BRANCH" : id == 10 ? "RVC" : id == 11 ? "ELW" : id == 12 ? "LD_EXT" : id == 13 ? "ST_EXT" : id == 14 ? "LD_EXT_CYC" : id == 15 ? "ST_EXT_CYC" : id == 16 ? "TCDM_CONT" : "NA")



// Performance event IDs (compatible with PMSIS)
#define PI_PERF_CYCLES          CSR_PCER_CYCLES
#define PI_PERF_INSTR           CSR_PCER_INSTR
#define PI_PERF_LD_STALL        CSR_PCER_LD_STALL
#define PI_PERF_JMP_STALL       CSR_PCER_JMP_STALL
#define PI_PERF_IMISS           CSR_PCER_IMISS
#define PI_PERF_LD              CSR_PCER_LD
#define PI_PERF_ST              CSR_PCER_ST
#define PI_PERF_JUMP            CSR_PCER_JUMP
#define PI_PERF_BRANCH          CSR_PCER_BRANCH
#define PI_PERF_TAKEN_BRANCH    CSR_PCER_TAKEN_BRANCH
#define PI_PERF_RVC             CSR_PCER_RVC
#define PI_PERF_LD_EXT          CSR_PCER_LD_EXT
#define PI_PERF_ST_EXT          CSR_PCER_ST_EXT
#define PI_PERF_LD_EXT_CYC      CSR_PCER_LD_EXT_CYC
#define PI_PERF_ST_EXT_CYC      CSR_PCER_ST_EXT_CYC
#define PI_PERF_TCDM_CONT       CSR_PCER_TCDM_CONT

// Benchmark statistics structure
typedef struct {
    unsigned int cycles;
    unsigned int instr;
    unsigned int ld;
    unsigned int st;
    unsigned int ld_stall;
    unsigned int jmp_stall;
    unsigned int imiss;
    unsigned int branch;
    unsigned int taken_branch;
    unsigned int rvc;
    unsigned int ld_ext;
    unsigned int st_ext;
    unsigned int ld_ext_cyc;
    unsigned int st_ext_cyc;
    unsigned int tcdm_cont;
} perf_stats_t;

// Initialize performance counters for comprehensive benchmarking
static inline void perf_bench_init() {
    // Enable all performance counters
    pi_perf_conf(
        (1 << PI_PERF_CYCLES) |
        (1 << PI_PERF_INSTR) |
        (1 << PI_PERF_LD_STALL) |
        (1 << PI_PERF_JMP_STALL) |
        (1 << PI_PERF_IMISS) |
        (1 << PI_PERF_LD) |
        (1 << PI_PERF_ST) |
        (1 << PI_PERF_JUMP) |
        (1 << PI_PERF_BRANCH) |
        (1 << PI_PERF_TAKEN_BRANCH) |
        (1 << PI_PERF_RVC) |
        (1 << PI_PERF_LD_EXT) |
        (1 << PI_PERF_ST_EXT) |
        (1 << PI_PERF_LD_EXT_CYC) |
        (1 << PI_PERF_ST_EXT_CYC) |
        (1 << PI_PERF_TCDM_CONT)
    );
}

// Start performance monitoring
static inline void perf_bench_start() {
    pi_perf_reset();
    pi_perf_start();
}

// Stop performance monitoring
static inline void perf_bench_stop() {
    pi_perf_stop();
}

// Read all performance counters into structure
static inline void perf_bench_read(perf_stats_t *stats) {
    stats->cycles = pi_perf_read(PI_PERF_CYCLES);
    stats->instr = pi_perf_read(PI_PERF_INSTR);
    stats->ld = pi_perf_read(PI_PERF_LD);
    stats->st = pi_perf_read(PI_PERF_ST);
    stats->ld_stall = pi_perf_read(PI_PERF_LD_STALL);
    stats->jmp_stall = pi_perf_read(PI_PERF_JMP_STALL);
    stats->imiss = pi_perf_read(PI_PERF_IMISS);
    stats->branch = pi_perf_read(PI_PERF_BRANCH);
    stats->taken_branch = pi_perf_read(PI_PERF_TAKEN_BRANCH);
    stats->rvc = pi_perf_read(PI_PERF_RVC);
    stats->ld_ext = pi_perf_read(PI_PERF_LD_EXT);
    stats->st_ext = pi_perf_read(PI_PERF_ST_EXT);
    stats->ld_ext_cyc = pi_perf_read(PI_PERF_LD_EXT_CYC);
    stats->st_ext_cyc = pi_perf_read(PI_PERF_ST_EXT_CYC);
    stats->tcdm_cont = pi_perf_read(PI_PERF_TCDM_CONT);
}

// Print performance statistics (core 0 only to avoid clutter)
static inline void perf_bench_print(const char *label, perf_stats_t *stats) {
    if (pi_core_id() == 0) {
        printf("\n=== Performance Statistics: %s ===\n", label);
        printf("Cycles:              %10u\n", stats->cycles);
        printf("Instructions:        %10u\n", stats->instr);
        printf("IPC:                 %10.3f\n",
               stats->cycles > 0 ? (float)stats->instr / stats->cycles : 0.0f);
        printf("\n--- Instruction Mix ---\n");
        printf("Loads:               %10u (%.2f%%)\n", stats->ld,
               stats->instr > 0 ? 100.0f * stats->ld / stats->instr : 0.0f);
        printf("Stores:              %10u (%.2f%%)\n", stats->st,
               stats->instr > 0 ? 100.0f * stats->st / stats->instr : 0.0f);
        printf("Branches:            %10u (%.2f%%)\n", stats->branch,
               stats->instr > 0 ? 100.0f * stats->branch / stats->instr : 0.0f);
        printf("Taken Branches:      %10u (%.2f%%)\n", stats->taken_branch,
               stats->branch > 0 ? 100.0f * stats->taken_branch / stats->branch : 0.0f);
        printf("Compressed (RVC):    %10u (%.2f%%)\n", stats->rvc,
               stats->instr > 0 ? 100.0f * stats->rvc / stats->instr : 0.0f);
        printf("\n--- Stalls & Hazards ---\n");
        printf("Load Stalls:         %10u\n", stats->ld_stall);
        printf("Jump Stalls:         %10u\n", stats->jmp_stall);
        printf("I-cache Misses:      %10u\n", stats->imiss);
        printf("TCDM Contentions:    %10u\n", stats->tcdm_cont);
        printf("\n--- Memory Hierarchy ---\n");
        printf("External Loads:      %10u (%.2f%%)\n", stats->ld_ext,
               stats->ld > 0 ? 100.0f * stats->ld_ext / stats->ld : 0.0f);
        printf("External Stores:     %10u (%.2f%%)\n", stats->st_ext,
               stats->st > 0 ? 100.0f * stats->st_ext / stats->st : 0.0f);
        printf("Ext Load Cycles:     %10u (avg: %.2f)\n", stats->ld_ext_cyc,
               stats->ld_ext > 0 ? (float)stats->ld_ext_cyc / stats->ld_ext : 0.0f);
        printf("Ext Store Cycles:    %10u (avg: %.2f)\n", stats->st_ext_cyc,
               stats->st_ext > 0 ? (float)stats->st_ext_cyc / stats->st_ext : 0.0f);
        printf("========================================\n\n");
    }
}

// Compute difference between two stats (for analyzing specific code sections)
static inline void perf_bench_diff(perf_stats_t *result,
                                    perf_stats_t *end,
                                    perf_stats_t *start) {
    result->cycles = end->cycles - start->cycles;
    result->instr = end->instr - start->instr;
    result->ld = end->ld - start->ld;
    result->st = end->st - start->st;
    result->ld_stall = end->ld_stall - start->ld_stall;
    result->jmp_stall = end->jmp_stall - start->jmp_stall;
    result->imiss = end->imiss - start->imiss;
    result->branch = end->branch - start->branch;
    result->taken_branch = end->taken_branch - start->taken_branch;
    result->rvc = end->rvc - start->rvc;
    result->ld_ext = end->ld_ext - start->ld_ext;
    result->st_ext = end->st_ext - start->st_ext;
    result->ld_ext_cyc = end->ld_ext_cyc - start->ld_ext_cyc;
    result->st_ext_cyc = end->st_ext_cyc - start->st_ext_cyc;
    result->tcdm_cont = end->tcdm_cont - start->tcdm_cont;
}

#endif // __PERF_UTILS_H__
