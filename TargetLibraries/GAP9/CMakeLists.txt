# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
  "$ENV{GAP_SDK_HOME}/tools/autotiler_v3/CNN_Libraries_fp32/CNN_Bias_Linear_Activation_fp32.c"
  "$ENV{GAP_SDK_HOME}/tools/autotiler_v3/CNN_Libraries/CNN_Copy.c"
)

# CNN_BasicKernels_NE16 from gap9-sdk redefines NE16_REG_* macros that
# pulp-nnx's ne16 hal also defines. For GAP9_w_NE16 we use the pulp-nnx
# NE16 stack; for plain GAP9 (Pu DENG's NE16-Linear path) we use the SDK's.
if(NOT platform STREQUAL "GAP9_w_NE16")
  list(APPEND SOURCES
    "$ENV{GAP_SDK_HOME}/tools/autotiler_v3/CNN_Libraries_HWPE/CNN_BasicKernels_NE16.c"
  )
endif()


# Exclude dory_mem and dory_dma from SOURCES (they need different optimization)
list(FILTER SOURCES EXCLUDE REGEX ".*dory_(mem|dma).*")

# RW: Include PULPOpen sources but exclude dory_mem related files
file(GLOB_RECURSE PULPOPEN_SOURCES "../PULPOpen/src/**")
list(FILTER PULPOPEN_SOURCES EXCLUDE REGEX ".*dory_mem.*")
list(APPEND SOURCES ${PULPOPEN_SOURCES})

# Separate dory library compiled without -O3
add_library(dory_lib STATIC
  ${CMAKE_CURRENT_LIST_DIR}/src/dory_mem.c
  ${CMAKE_CURRENT_LIST_DIR}/src/dory_dma.c
)
target_include_directories(dory_lib PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/inc
)
target_compile_options(dory_lib PRIVATE
  -Wno-implicit-function-declaration
  -Wno-sign-conversion
  -Wno-sign-compare
  -Wno-type-limits
  -Wno-attributes
  -Wno-incompatible-pointer-types
  -Og
)
target_compile_definitions(dory_lib PUBLIC NUM_CORES=${NUM_CORES})
target_link_libraries(dory_lib PUBLIC pmsis)

add_deeploy_library(deeploygap9 STATIC ${SOURCES})
target_include_directories(deeploygap9
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/inc
  ${TILER_INC}
  ${TILER_EMU_INC}
  ${TILER_CNN_KERNEL_PATH_FP32}
  ${TILER_CNN_KERNEL_PATH_FP16}
  $ENV{GAP_SDK_HOME}/tools/autotiler_v3/CNN_Libraries_SQ8
  $ENV{GAP_SDK_HOME}/tools/autotiler_v3/CNN_Libraries
  ${TILER_DSP_KERNEL_V2_PATH}
  ${TILER_DSP_KERNEL_V2_PATH}/FastMathFunctions
)


target_compile_options(deeploygap9 PUBLIC
    -DNUM_CORES=${NUM_CORES}
    -DSTD_FLOAT
  )

target_compile_options(deeploygap9 PRIVATE
  -Wno-sign-conversion
  -Wno-sign-compare
  -Wno-type-limits
  -Wno-attributes
  -Wno-incompatible-pointer-types
  -O3
)

#RW: Link PULP-NN
#RW: Set PULP-NN version and bitwidth for pulp-nn-mixed
set(PULPNNVERSION XPULPV2)
set(PULPNNBITWIDTH 32)

# Option to use prebuilt pulp-nn library (default: ON for faster builds)
option(USE_PREBUILT_PULPNN "Use prebuilt pulp-nn-mixed librar with 8 cores" ON)

# Check if NUM_CORES matches prebuilt library requirement
if(USE_PREBUILT_PULPNN AND NOT NUM_CORES EQUAL 8)
  message(WARNING "[Deeploy GAP9] Prebuilt pulp-nn library is compiled for 8 cores, but NUM_CORES=${NUM_CORES}.")
  message(WARNING "[Deeploy GAP9] Disabling prebuilt library and building from source instead.")
  set(USE_PREBUILT_PULPNN OFF)
endif()

if(USE_PREBUILT_PULPNN)
  # Use prebuilt pulp-nn library
  set(PREBUILT_PULPNN_DIR ${CMAKE_CURRENT_LIST_DIR}/prebuilt)

  if(EXISTS ${PREBUILT_PULPNN_DIR}/libpulp-nn-mixed.a)
    message(STATUS "[Deeploy GAP9] Using prebuilt pulp-nn-mixed library")

    # Create imported target for prebuilt library
    add_library(pulp-nn-mixed STATIC IMPORTED)
    set_target_properties(pulp-nn-mixed PROPERTIES
      IMPORTED_LOCATION ${PREBUILT_PULPNN_DIR}/libpulp-nn-mixed.a
      INTERFACE_INCLUDE_DIRECTORIES ${PREBUILT_PULPNN_DIR}/include
    )
  else()
    message(WARNING "[Deeploy GAP9] Prebuilt pulp-nn library not found at ${PREBUILT_PULPNN_DIR}")
    message(WARNING "[Deeploy GAP9] Falling back to building from source")
    set(USE_PREBUILT_PULPNN OFF)
  endif()
endif()

if(NOT USE_PREBUILT_PULPNN)
  # Build pulp-nn from source
  message(STATUS "[Deeploy GAP9] Building pulp-nn-mixed from source")

  add_subdirectory(../third_party/pulp-nn-mixed ${CMAKE_CURRENT_BINARY_DIR}/pulp-nn-mixed)

  #RW: GCC not recognizing -Wno-typedef-redefinition defined in PULP-NN CMakelist
  target_compile_options(pulp-nn-mixed PRIVATE -Wno-error)
  target_compile_definitions(pulp-nn-mixed PUBLIC NUM_CORES=${NUM_CORES})
  target_link_libraries(pulp-nn-mixed PUBLIC pmsis)
endif()

target_link_libraries(deeploygap9 PUBLIC pulp-nn-mixed)

# NE16 accelerator (via pulp-nnx) for GAP9_w_NE16 platform
if(platform STREQUAL "GAP9_w_NE16")
  set(USE_NE16 ON CACHE BOOL "Use the NE16 accelerator." FORCE)
  add_subdirectory(../third_party/pulp-nnx ${CMAKE_CURRENT_BINARY_DIR}/pulp-nnx)
  target_link_libraries(pulp-nnx PUBLIC pmsis)
  target_compile_options(pulp-nnx PRIVATE
    -Wno-error
    -Wno-implicit-int-conversion
    -Wno-sign-conversion
    -Wno-typedef-redefinition
    -Wno-unused-parameter
    -Wno-incompatible-pointer-types-discards-qualifiers
  )
  target_link_libraries(deeploygap9 PUBLIC pulp-nnx)
endif()

target_link_libraries(deeploygap9 PUBLIC pmsis)
target_link_libraries(deeploygap9 PUBLIC m)
target_link_libraries(deeploygap9 PUBLIC dory_lib)
