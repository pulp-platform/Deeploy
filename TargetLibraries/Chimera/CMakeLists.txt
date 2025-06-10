file(GLOB_RECURSE SOURCES
  "src/**"
)

# Import chimera-open target
set(TARGET_PLATFORM "chimera-open")
set(TEST_MODE "none")
add_subdirectory(
  /app/Deeploy/install/chimera-sdk
  ${CMAKE_BINARY_DIR}/chimera-open
)

add_deeploy_library(deeploychimera STATIC ${SOURCES})
target_include_directories(deeploychimera
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${CMAKE_CURRENT_LIST_DIR}/inc/kernel
)

target_link_libraries(deeploychimera INTERFACE runtime_host hal_host)