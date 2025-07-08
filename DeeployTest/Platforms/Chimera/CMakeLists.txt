set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)

add_gvsoc_emulation(${ProjectId} "chimera")

link_compile_dump(${TESTNAME})