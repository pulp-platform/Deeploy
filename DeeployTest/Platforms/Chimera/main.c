/*
 * Copyright 2025 ETH Zurich.
 * Licensed under the Apache License, Version 2.0, see LICENSE for details.
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Victor Jung <jungvi@iis.ee.ethz.ch>
 */

#include <stdio.h>

#include "soc.h"
#include "driver.h"
#include "uart.h"

int main() {

    volatile int32_t a = 42;

    printf("BONJOUR ZURICH\n");

    return 0;
}