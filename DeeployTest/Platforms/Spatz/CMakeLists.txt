set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)

# Add QuestaSim RTL simulation for Spatz (mirroring Snitch's vsim)
add_spatz_vsim_simulation(${ProjectId})
