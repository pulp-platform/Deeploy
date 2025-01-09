file(GLOB_RECURSE SOURCES
  "src/**"
)

include(cmake/snitch-runtime-precompiled.cmake)

add_deeploy_library(deeploysnitch STATIC ${SOURCES})
target_include_directories(deeploysnitch
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${CMAKE_CURRENT_LIST_DIR}/inc/kernel
)


target_include_directories(deeploysnitch SYSTEM PUBLIC ${SNITCH_RUNTIME_INCLUDE})
target_compile_options(deeploysnitch PUBLIC ${SNITCH_RUNTIME_COMPILE_FLAGS})

target_link_libraries(deeploysnitch INTERFACE snitch-runtime)
