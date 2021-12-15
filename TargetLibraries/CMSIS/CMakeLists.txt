file(GLOB_RECURSE SOURCES
  "src/**"
)

add_deeploy_library(deeploycmsis STATIC ${SOURCES})

target_include_directories(deeploycmsis
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
)

# SCHEREMO: CMSIS-NN
add_subdirectory(third_party/CMSIS-NN)
target_compile_options(cmsis-nn
  PRIVATE
  -DARM_MATH_DSP
  -DARM_MATH_LOOPUNROLL
  -Ofast
  # SCHEREMO: Waive CMSIS-NN warnings
  -Wno-sign-conversion
  -Wno-conversion
)

target_link_libraries(deeploycmsis INTERFACE cmsis-nn)
