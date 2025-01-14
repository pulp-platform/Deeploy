set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
add_banshee_simulation(${ProjectId})
add_gvsoc_simulation(${ProjectId})
add_snitch_cluster_vsim_simulation(${ProjectId})
add_snitch_cluster_vsim_gui_simulation(${ProjectId})