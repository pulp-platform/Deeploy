// Copyright 2025 University of Bologna and Fondazione Chips-IT.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Alberto Dequino <alberto.dequino@unibo.it>
// Alex Marchioni <alex.marchioni@chips.it>

#include <stdint.h>
#include "testinputs.h"
#include "testoutputs.h"
#include "Network.h"



int main(void) {

    uint32_t hartid = get_hartid();
    uint32_t l1_tile_base = get_l1_base(hartid);
    uint32_t cycle_start, cycle_stop;

    /* Init tile's iDMA, Redmule, fsync, event-unit */
    idma_config_t idma_cfg = {.hartid = hartid};
    idma_controller_t idma_ctrl = {
        .base = NULL,
        .cfg = &idma_cfg,
        .api = &idma_api,
    };

    redmule_config_t redmule_cfg = {.hartid = hartid};
    redmule_controller_t redmule_ctrl = {
        .base = NULL,
        .cfg = &redmule_cfg,
        .api = &redmule_api,
    };

    fsync_config_t fsync_cfg = {.hartid = hartid};
    fsync_controller_t fsync_ctrl = {
        .base = NULL,
        .cfg = &fsync_cfg,
        .api = &fsync_api,
    };

    fsync_init(&fsync_ctrl);
    idma_init(&idma_ctrl);
    redmule_init(&redmule_ctrl);

    eu_config_t eu_cfg = {.hartid = hartid};
    eu_controller_t eu_ctrl = {
        .base = NULL,
        .cfg = &eu_cfg,
        .api = &eu_api,
    };
    eu_init(&eu_ctrl);
    eu_fsync_init(&eu_ctrl, 0);
    eu_redmule_init(&eu_ctrl, 0);
    eu_idma_init(&eu_ctrl, 0);


    /* initialization */
    InitNetwork();

    fsync_sync_level(&fsync_ctrl, MAX_SYNC_LVL - 1, 0);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* input copy */
    // TODO: check if memcopy is necessary!!!
    if (hartid == 0) {
        for (uint32_t buf = 0; buf < num_inputs; buf++) {
            memcpy(inputs[buf], _inputs[buf], inputs_bytes[buf]);
        }
    }

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* execution */
    cycle_start = perf_get_cycles();
    RunNetwork();
    cycle_stop = perf_get_cycles();
    printf("id: %d, cycles: %d\n", hartid, cycle_stop - cycle_start);

    fsync_sync_global(&fsync_ctrl);
    eu_fsync_wait(&eu_ctrl, WAIT_MODE);

    /* comparison */
    uint32_t errors = 0;
    uint32_t tests = 0;
    OUTPUTTYPE *computed_buf;
    OUTPUTTYPE *expected_buf;
    if (hartid == 0) {
        for (uint32_t buf = 0; buf < num_outputs; buf++) {
            tests += outputs_bytes[buf] / sizeof(OUTPUTTYPE);
            for (uint32_t i = 0; i < outputs_bytes[buf] / sizeof(OUTPUTTYPE); i++) {
                OUTPUTTYPE expected = ((OUTPUTTYPE *)_outputs[buf])[i];
                OUTPUTTYPE computed = ((OUTPUTTYPE *)outputs[buf])[i];
                OUTPUTTYPE diff = (computed > expected) ? (computed - expected) : (expected - computed);
                if(diff > 0) {
                    if (ISOUTPUTFLOAT) {
                        // printf("Expected %10.6f computed: %10.6f diff: %10.6f (at index %u in output %u)\n",
                        //     expected, computed, diff, i, buf);
                    } else {
                        printf("Expected %d computed: %d diff: %d (at index %u in output %u)\n",
                            expected, computed, diff, i, buf);
                    }
                    errors++;
                }
            }
        }
        printf("Number of errors: %d\n", errors);
    }

    return errors;
}