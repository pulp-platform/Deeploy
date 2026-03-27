set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)

add_spatz_gvsoc_emulation(${ProjectId} "spatz_v2")
add_spatz_vsim_simulation(${ProjectId})