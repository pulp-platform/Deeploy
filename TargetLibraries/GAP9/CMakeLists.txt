# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

file(GLOB_RECURSE SOURCES
  "src/**"
)

# RW: Include PULPOpen sources but exclude dory_mem related files
file(GLOB_RECURSE PULPOPEN_SOURCES "../PULPOpen/src/**")
list(FILTER PULPOPEN_SOURCES EXCLUDE REGEX ".*dory_mem.*")
list(APPEND SOURCES ${PULPOPEN_SOURCES})

add_deeploy_library(deeploygap9 STATIC ${SOURCES})
target_include_directories(deeploygap9
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
  ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/inc
)

target_compile_options(deeploygap9 PUBLIC
    -DNUM_CORES=${NUM_CORES}
  )

target_compile_options(deeploygap9 PRIVATE
  -Wno-sign-conversion
  -Wno-sign-compare
  -Wno-type-limits
  -Wno-attributes
)

target_link_libraries(deeploygap9 PUBLIC pmsis)

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

# Add pulp-trainlib integration for training/gradient operators
set(PULP_TRAINLIB_DIR ${CMAKE_CURRENT_LIST_DIR}/../PULPOpen/third_party/pulp-trainlib)

target_include_directories(deeploygap9 PUBLIC
  ${PULP_TRAINLIB_DIR}/lib/include
)

file(GLOB PULP_TRAINLIB_CONV_SOURCES
  "${PULP_TRAINLIB_DIR}/lib/sources/pulp_conv2d_fp32.c"
  "${PULP_TRAINLIB_DIR}/lib/sources/pulp_im2col_fp32.c"
  "${PULP_TRAINLIB_DIR}/lib/sources/pulp_conv_dw_fp32.c"
  "${PULP_TRAINLIB_DIR}/lib/sources/pulp_conv_pw_fp32.c"
  "${PULP_TRAINLIB_DIR}/lib/sources/pulp_matmul_fp32.c"
  "${PULP_TRAINLIB_DIR}/lib/sources/pulp_train_utils_fp32.c"
  "${PULP_TRAINLIB_DIR}/lib/sources/pulp_conv_naive_fp32.c"
)
target_sources(deeploygap9 PRIVATE ${PULP_TRAINLIB_CONV_SOURCES})

set_source_files_properties(${PULP_TRAINLIB_CONV_SOURCES}
  PROPERTIES
  COMPILE_FLAGS "-UUSE_DMA -Dfloat16alt=float -fno-strict-aliasing"
)

target_link_libraries(deeploygap9 PUBLIC m)

