# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0


set(SPATZ_RUNTIME_BASE_INCLUDE
	${SPATZ_HOME}/sw/snRuntime/include
	${SPATZ_HOME}/sw/snRuntime/vendor
	${SPATZ_HOME}/sw/toolchain/riscv-opcodes
)

set(SPATZ_CLUSTER_LINK_INCLUDE
	${SPATZ_HOME}/hw/system/spatz_cluster/sw/build/snRuntime
)

set(SPATZ_LINKER_SCRIPT ${SPATZ_HOME}/hw/system/spatz_cluster/sw/build/snRuntime/common.ld)
# set(SPATZ_LINKER_SCRIPT ${SNITCH_RUNTIME_HOME}/base.ld)
if(NOT EXISTS ${SPATZ_LINKER_SCRIPT})
	message(FATAL_ERROR "Spatz linker script not found: ${SPATZ_LINKER_SCRIPT}")
endif()

set(SPATZ_CLUSTER_LINK_OPTIONS
	-Wl,--gc-sections
	-T ${SPATZ_LINKER_SCRIPT}
)

set(SPATZ_RUNTIME_INCLUDE ${SPATZ_RUNTIME_BASE_INCLUDE})

add_library(spatz-runtime INTERFACE)
target_link_directories(spatz-runtime INTERFACE ${SPATZ_CLUSTER_LINK_INCLUDE})
target_link_libraries(spatz-runtime INTERFACE ${SPATZ_CLUSTER_LINK_OPTIONS} libsnRuntime-cluster.a)
