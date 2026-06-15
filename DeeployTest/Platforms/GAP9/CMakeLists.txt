# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})



set(${SDKCONFIG_FILE} ${CMAKE_CURRENT_LIST_DIR}/sdk.config)

option(TRAINING "Use training harness instead of inference harness" OFF)

# Compile-time training parameters (override via -D on cmake command line)
set(N_TRAIN_STEPS "1" CACHE STRING "Number of optimizer steps")
set(N_ACCUM_STEPS "1" CACHE STRING "Number of mini-batches per optimizer step")
set(TRAINING_NUM_DATA_INPUTS "2" CACHE STRING "Number of data inputs per mini-batch")

if(TRAINING)
    file(GLOB_RECURSE SOURCES
        src/CycleCounter.c
        src/deeploytraintest.c
    )
    set(NETWORK_LIB training_network)
else()
    file(GLOB_RECURSE SOURCES
        src/CycleCounter.c
        src/deeploytest.c
    )
    set(NETWORK_LIB network)
endif()

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
# add_executable(${ProjectId} ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

if(TRAINING)
    target_link_libraries(${ProjectId} PRIVATE ${NETWORK_LIB} optimizer_network deeploylib)
else()
    target_link_libraries(${ProjectId} PRIVATE ${NETWORK_LIB} deeploylib)
endif()
target_compile_options(${ProjectId} INTERFACE ${NETWORK_LIB})

if(TRAINING)
    target_compile_definitions(${ProjectId} PRIVATE
        N_TRAIN_STEPS=${N_TRAIN_STEPS}
        N_ACCUM_STEPS=${N_ACCUM_STEPS}
        TRAINING_NUM_DATA_INPUTS=${TRAINING_NUM_DATA_INPUTS}
    )
endif()

add_gvsoc_emulation(${ProjectId} "gap9.evk")
add_board_deployment(${ProjectId} "gap9.evk")

if(POWER_MEASUREMENT)
  target_compile_definitions(${ProjectId} PRIVATE POWER_MEASUREMENT)
endif()

if(USE_SEQUENTIAL_RADEMACHER)
  target_compile_definitions(${NETWORK_LIB} PRIVATE USE_SEQUENTIAL_RADEMACHER)
endif()

# RW: Waive sign comparison warnings from pulp_nn_utils.h
target_compile_options(${NETWORK_LIB} PRIVATE
    -Wno-sign-compare
    -Wno-pointer-sign
    -Wno-unknown-pragmas
    -Wno-error
  )

target_link_options(${ProjectId} PRIVATE
    -Wl,--print-memory-usage
)
link_compile_dump(${TESTNAME})
