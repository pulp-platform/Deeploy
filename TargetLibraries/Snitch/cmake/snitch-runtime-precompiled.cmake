# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

set(SNITCH_RUNTIME_BASE_INCLUDE
  ${SNITCH_RUNTIME_HOME}/src
  ${SNITCH_RUNTIME_HOME}/api
  ${SNITCH_RUNTIME_HOME}/impl
  ${SNITCH_RUNTIME_HOME}/../deps/riscv-opcodes # TODO: generate riscv-opcodes whatever
)

set(SNITCH_RUNTIME_OMP_INCLUDE
  ${SNITCH_RUNTIME_HOME}/src/omp
  ${SNITCH_RUNTIME_HOME}/api/omp
)

set(SNITCH_CLUSTER_LINK_OPTIONS
  -Wl,--gc-sections
  -T ${SNITCH_RUNTIME_HOME}/base.ld
)

set(SNITCH_RUNTIME_INCLUDE ${SNITCH_RUNTIME_BASE_INCLUDE} ${SNITCH_RUNTIME_OMP_INCLUDE})


add_library(snitch-runtime INTERFACE)
target_link_directories(snitch-runtime INTERFACE ${SNITCH_RUNTIME_HOME}/build ${SNITCH_RUNTIME_HOME}/impl)
target_link_libraries(snitch-runtime INTERFACE ${SNITCH_CLUSTER_LINK_OPTIONS} libsnRuntime.a)