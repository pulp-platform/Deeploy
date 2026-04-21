set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

list(APPEND SOURCES
    ${SPATZ_HOME}/sw/spatzBenchmarks/benchmark/benchmark.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})

set(SPATZ_BENCHMARK_INCLUDE_DIR
    ${SPATZ_HOME}/sw/spatzBenchmarks/include
)
target_include_directories(${ProjectId} PRIVATE ${SPATZ_BENCHMARK_INCLUDE_DIR})

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)

add_spatz_gvsoc_emulation(${ProjectId} "spatz_v2")
add_spatz_vsim_simulation(${ProjectId})