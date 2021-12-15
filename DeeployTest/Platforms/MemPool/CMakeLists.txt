set(ProjectId ${TESTNAME})

file(GLOB_RECURSE SOURCES
    main.c
)

link_directories(${ProjectId}/../../${GENERATED_SOURCE})

add_deeploy_executable(${ProjectId} EXCLUDE_FROM_ALL ${SOURCES} )
add_dependencies(${ProjectId} linkerscript)

target_link_libraries(${ProjectId} PRIVATE network deeploylib)
add_banshee_simulation(${ProjectId})

link_compile_dump(${TESTNAME})
