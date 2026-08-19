# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

set_source_files_properties(${SOURCES} PROPERTIES LANGUAGE CXX)
add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
if(DEFINED ENV{BANSHEE_INSTALL_DIR})
    add_banshee_simulation(${ProjectId})
endif()
add_gvsoc_emulation(${ProjectId} "pulp.snitch.snitch_cluster_single")
add_snitch_cluster_vsim_simulation(${ProjectId})
add_snitch_cluster_vsim_gui_simulation(${ProjectId})
add_snitch_cluster_verilator_simulation(${ProjectId})