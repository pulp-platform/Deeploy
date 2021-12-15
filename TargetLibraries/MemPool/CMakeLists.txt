file(GLOB_RECURSE SOURCES
  "src/**"
)

include(cmake/mempool-runtime.cmake)

add_deeploy_library(deeploymempool STATIC ${SOURCES})

target_include_directories(deeploymempool
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_include_directories(deeploymempool PUBLIC
  ${MEMPOOL_RUNTIME_INCLUDE}
  ${MEMPOOL_RUNTIME_OMP_INCLUDE}
  ${MEMPOOL_RUNTIME_HALIDE_INCLUDE}
)
target_compile_options(deeploymempool PUBLIC ${MEMPOOL_RUNTIME_COMPILE_FLAGS})

target_link_libraries(deeploymempool INTERFACE mempool-runtime)
target_sources(deeploymempool INTERFACE $<TARGET_OBJECTS:mempool-runtime>)
