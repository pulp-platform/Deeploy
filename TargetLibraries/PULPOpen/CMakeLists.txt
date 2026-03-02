# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
)

if(NOT DEFINED ENV{PULP_SDK_HOME})
  message(FATAL_ERROR "Environment variable PULP_SDK_HOME not set.")
endif()

if(platform STREQUAL "Siracusa" OR platform STREQUAL "Siracusa_w_neureka")
  include(cmake/pulp-sdk-siracusa.cmake)
elseif(platform STREQUAL "PULPOpen")
  include(cmake/pulp-sdk-pulp-open.cmake)
else()
  message(FATAL_ERROR "Unrecognized platform: ${platform}")
endif()

add_deeploy_library(deeploypulp STATIC ${SOURCES})
target_include_directories(deeploypulp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_compile_options(deeploypulp PRIVATE
  -Wno-implicit-function-declaration
  -Wno-implicit-int-conversion
  -Wno-sign-conversion
  -Wno-sign-compare
  -Wno-typedef-redefinition
  -Wno-unused-parameter
  -Wno-unused-function
  -Wno-unused-variable
  -Wno-uninitialized
)

target_include_directories(deeploypulp PUBLIC ${PULP_SDK_INCLUDES})
target_compile_options(deeploypulp PUBLIC ${PULP_SDK_COMPILE_FLAGS})

add_subdirectory(../third_party/pulp-nn-mixed ${CMAKE_CURRENT_BINARY_DIR}/pulp-nn-mixed)
target_include_directories(pulp-nn-mixed PUBLIC ${PULP_SDK_INCLUDES})
target_compile_options(pulp-nn-mixed PUBLIC ${PULP_SDK_COMPILE_FLAGS})

# Add pulp-trainlib integration
target_include_directories(deeploypulp PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/include
)

# Add necessary pulp-trainlib sources for ConvGradW
file(GLOB PULP_TRAINLIB_CONV_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv2d_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_im2col_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_dw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_pw_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_matmul_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_train_utils_fp32.c"
  "${CMAKE_CURRENT_LIST_DIR}/third_party/pulp-trainlib/lib/sources/pulp_conv_naive_fp32.c"
)
target_sources(deeploypulp PRIVATE ${PULP_TRAINLIB_CONV_SOURCES})

# Fix compilation issues with pulp-trainlib sources
# Undefine USE_DMA and define float16alt to avoid conflicts
set_source_files_properties(${PULP_TRAINLIB_CONV_SOURCES}
  PROPERTIES
  COMPILE_FLAGS "-UUSE_DMA -Dfloat16alt=float"
)

target_link_libraries(deeploypulp PUBLIC pulp-nn-mixed)
target_link_libraries(deeploypulp INTERFACE pulp-sdk)
target_sources(deeploypulp INTERFACE $<TARGET_OBJECTS:pulp-sdk>)

set(PULP_NNX_PLATFORMS "Siracusa;Siracusa_w_neureka")
if (platform IN_LIST PULP_NNX_PLATFORMS)
  if (platform STREQUAL "Siracusa" OR platform STREQUAL "Siracusa_w_neureka")
    set(USE_NEUREKA ON)
  else()
    message(FATAL_ERROR "Missing accelerator flags for platform ${platform}")
  endif()
  add_subdirectory(../third_party/pulp-nnx ${CMAKE_CURRENT_BINARY_DIR}/pulp-nnx)
  target_include_directories(pulp-nnx PUBLIC ${PULP_SDK_INCLUDES})
  target_compile_options(pulp-nnx PUBLIC ${PULP_SDK_COMPILE_FLAGS})
    target_compile_options(pulp-nnx PRIVATE
    -Wno-implicit-int-conversion
    -Wno-sign-conversion
    -Wno-typedef-redefinition
    -Wno-unused-parameter
    -Wno-incompatible-pointer-types-discards-qualifiers
    )
  target_link_libraries(deeploypulp INTERFACE pulp-nnx)
endif()
