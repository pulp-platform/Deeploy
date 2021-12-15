set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    src/CycleCounter.c
    src/deeploytest.c
)

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES})
target_include_directories(${ProjectId} PRIVATE ${CMAKE_CURRENT_LIST_DIR}/inc)

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
target_compile_options(${ProjectId} INTERFACE network)
add_gvsoc_emulation(${ProjectId})

link_compile_dump(${TESTNAME})
