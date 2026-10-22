// Copyright 2023 ETH Zurich and University of Bologna.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
//
// Bare-metal offload test for the PULP cluster

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#include "car_memory_map.h"
#include "car_util.h"
#include "dif/clint.h"
#include "dif/uart.h"
#include "params.h"
#include "regs/cheshire.h"
#include "util.h"
#include "payload.h"
#include "printf.h"

int main(void) {
    // Initializing Carfield (PULP domain too).
    car_init_start();
    car_reset_domain(CAR_PULP_RST);
    // Initializing PULP domain.
    //car_enable_domain (CAR_PULP_RST);

    uint32_t rtc_freq = *reg32(&__base_regs, CHESHIRE_RTC_FREQ_REG_OFFSET);
    uint64_t reset_freq = clint_get_core_freq(rtc_freq, 2500);

    // UART initialization for printing the results.
    uart_init(&__base_uart, reset_freq, 115200);

    // Binary loading
    load_binary();

    // The `load_binary()` function is usually loading the L2 or LLC with the
    // cluster code. Both L2 and LLC are in the cachable region, and since we
    // use WB caches we need to flush them to commit the content to the target
    // memories.
    fencei();

    volatile uint32_t pulp_boot_default = ELF_BOOT_ADDR;
    volatile uint32_t pulp_ret_val = 0;

    pulp_cluster_set_bootaddress(pulp_boot_default);
    printf("Cluster boot.\r\n");

    // Start of end-to-end measurement of the offload test.
    uint64_t cyc_start = get_mcycle();

    pulp_cluster_start();
    pulp_cluster_wait_eoc();

    // End of end-to-end measurement of the offload test.
    uint64_t cyc_end = get_mcycle();

    // Bit 31 is set by the cluster runtime as a "return value valid" flag, so it can be masked
    pulp_ret_val = pulp_cluster_get_return() & 0x7FFFFFFF;

    // Results printing.
    printf("Errors(PULP return value): %u\r\n", pulp_ret_val);
    printf("RTC freq: %u Hz\r\n", rtc_freq);
    printf("CVA6 core freq: %lu Hz\r\n", reset_freq);
    printf("Offloading end-to-end cycles: %lu\r\n", cyc_end - cyc_start);

    pulp_cluster_stop();

    car_reset_domain(CAR_PULP_RST);

    return pulp_ret_val;
}
