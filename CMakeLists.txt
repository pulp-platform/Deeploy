# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.19)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_STANDARD 99)

set(CMAKE_C_COMPILER_LAUNCHER "ccache")
set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

if(TOOLCHAIN STREQUAL GCC)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(platform MemPool CACHE STRING "Platform (MemPool, SoftHier, QEMU, Siracusa, Siracusa_w_neureka, PULP-Open, GAP9, Generic, Snitch)")
set_property(CACHE platform PROPERTY STRINGS MemPool SoftHier QEMU Siracusa Siracusa_w_neureka PULP-Open GAP9 Generic Snitch)

if(platform STREQUAL MemPool)
  message(STATUS "Building for platform 'MemPool'")
elseif(platform STREQUAL QEMU-ARM)
  message(STATUS "Building for platform 'QEMU-ARM'")
elseif(platform STREQUAL Siracusa)
  message(STATUS "Building for platform 'Siracusa'")
elseif(platform STREQUAL Siracusa_w_neureka)
  message(STATUS "Building for platform 'Siracusa_w_neureka'")
elseif(platform STREQUAL PULPOpen)
  message(STATUS "Building for platform 'PULP-Open'")
elseif(platform STREQUAL GAP9)
  message(STATUS "Building for platform 'GAP9'")
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
  set(ENV{KCONFIG_CONFIG} DeeployTest/Platforms/GAP9/sdk.config)
  include($ENV{GAP_SDK_HOME}/utils/cmake/setup.cmake)
elseif(platform STREQUAL Generic)
  message(STATUS "Building for platform 'Generic'")
elseif(platform STREQUAL Snitch)
  message(STATUS "Building for platform 'Snitch'")
elseif(platform STREQUAL SoftHier)
  message(STATUS "Building for platform 'SoftHier'")
elseif(platform STREQUAL Chimera)
  message(STATUS "Building for platform 'Chimera'")
else()
  message(FATAL_ERROR "Invalid platform '${platform}' specified!")
endif()

# Import useful functions / macros
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Util.cmake)
# Only if not GAP9
if(NOT platform STREQUAL GAP9)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake)
endif()
include(${CMAKE_CURRENT_LIST_DIR}/cmake/simulation.cmake)

add_library(deeploylib INTERFACE)

message(STATUS "============================= Project Configuration ============================")
message(STATUS "[Deeploy]     platform               = " ${platform})
message(STATUS "[Deeploy]     use_dma                = " ${use_dma})
message(STATUS "================================================================================")
message(STATUS "")

if(platform STREQUAL MemPool)
  set(mempool_flavour mempool_ita CACHE STRING "Platform (mempool, mempool_ita or minpool)")
  set_property(CACHE mempool_flavour PROPERTY STRINGS mempool minpool mempool_ita)

  if(TOOLCHAIN STREQUAL LLVM)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/mempool/toolchain_llvm.cmake)
  else()
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/mempool/toolchain_gcc.cmake)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/mempool/${mempool_flavour}.cmake)

  project(deeploy LANGUAGES C ASM)

  add_subdirectory(TargetLibraries/MemPool)
  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(DeeployTest)
  target_include_directories(deeploymempool PUBLIC TargetLibraries/Generic/inc)

  target_link_libraries(deeploylib INTERFACE deeploymempool deeploybasic)

  message(STATUS "============================= MemPool Configuration ============================")
  message(STATUS "[cMake  ]   mempool_flavour        = " ${mempool_flavour})
  message(STATUS "[cMake  ]   boot_addr              = " ${boot_addr})
  message(STATUS "[cMake  ]   l2_base                = " ${l2_base})
  message(STATUS "[cMake  ]   l2_size                = " ${l2_size})
  message(STATUS "[cMake  ]   l2_banks               = " ${l2_banks})
  message(STATUS "[cMake  ]   seq_mem_size           = " ${seq_mem_size})
  message(STATUS "[cMake  ]   stack_size             = " ${stack_size})
  message(STATUS "[cMake  ]   axi_data_width         = " ${axi_data_width})
  message(STATUS "[cMake  ]   ro_line_width          = " ${ro_line_width})
  message(STATUS "[cMake  ]   dmas_per_group         = " ${dmas_per_group})
  message(STATUS "[cMake  ]   xqueue_size            = " ${xqueue_size})
  message(STATUS "[cMake  ]   xpulpimg               = " ${xpulpimg})
  message(STATUS "[cMake  ]   num_cores              = " ${num_cores})
  message(STATUS "[cMake  ]   num_eff_cores          = " ${num_eff_cores})
  message(STATUS "[cMake  ]   num_groups             = " ${num_groups})
  message(STATUS "[cMake  ]   num_cores_per_tile     = " ${num_cores_per_tile})
  message(STATUS "[cMake  ]   banking_factor         = " ${banking_factor})
  message(STATUS "[cMake  ]   axi_hier_radix         = " ${axi_hier_radix})
  message(STATUS "[cMake  ]   axi_masters_per_group  = " ${axi_masters_per_group})
  if(mempool_flavour STREQUAL mempool_ita)
    message(STATUS "=============================== ITA Configuration ==============================")
    message(STATUS "[cMake  ]   ita_pe                 = " ${ita_pe})
  endif()
  message(STATUS "================================================================================")
  message(STATUS "")

endif()

if(platform STREQUAL SoftHier)

  if(TOOLCHAIN STREQUAL GCC)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/softhier/toolchain_gcc.cmake)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/softhier/softhier_gvsoc.cmake)

  project(deeploy LANGUAGES C ASM)

  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(TargetLibraries/SoftHier)
  target_include_directories(deeploysofthier PUBLIC TargetLibraries/Generic/inc)

  add_subdirectory(DeeployTest)

  target_link_libraries(deeploylib INTERFACE deeploysofthier)

endif()

if(platform STREQUAL Generic)

  if(TOOLCHAIN STREQUAL LLVM)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/generic/toolchain_llvm.cmake)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/generic/generic.cmake)

  project(deeploy LANGUAGES C ASM)

  message(STATUS "============================= Generic Configuration ============================")
  message(STATUS "[cMake  ]   CPU                    = " ${CMAKE_SYSTEM_PROCESSOR})
  message(STATUS "================================================================================")
  message(STATUS "")

  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(DeeployTest)

  target_link_libraries(deeploylib INTERFACE deeploybasic)

endif()

if(platform STREQUAL QEMU-ARM)

  if(TOOLCHAIN STREQUAL LLVM)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/cmsis/toolchain_llvm.cmake)
  else()
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/cmsis/toolchain_gcc.cmake)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/cmsis/cmsis.cmake)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/cmsis/qemu.cmake)

  project(deeploy LANGUAGES C ASM)

  message(STATUS "============================= QEMU Configuration ============================")
  message(STATUS "[cMake  ]   CPU                    = " ${CPU})
  message(STATUS "[cMake  ]   FABI                   = " ${FABI})
  message(STATUS "[cMake  ]   FPU                    = " ${FPU})
  message(STATUS "================================================================================")
  message(STATUS "")

  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(TargetLibraries/CMSIS)
  add_subdirectory(DeeployTest)

  target_include_directories(deeploycmsis PUBLIC TargetLibraries/Generic/inc)
  target_link_libraries(deeploylib INTERFACE deeploybasic deeploycmsis)

endif()

if(platform STREQUAL Siracusa OR platform STREQUAL Siracusa_w_neureka OR platform STREQUAL PULPOpen)

  if(TOOLCHAIN STREQUAL LLVM)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/pulp/toolchain_llvm.cmake)
  else()
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/pulp/toolchain_gcc.cmake)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/pulp/pulp.cmake)

  if(platform STREQUAL Siracusa OR platform STREQUAL Siracusa_w_neureka)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/pulp/siracusa/siracusa.cmake)
  elseif(platform STREQUAL PULPOpen)
    include(${CMAKE_CURRENT_LIST_DIR}/cmake/pulp/pulp-open/pulp-open.cmake)
  endif()

  project(deeploy LANGUAGES C ASM)

  message(STATUS "============================= ${platform} Configuration ============================")
  message(STATUS "[cMake  ]   ISA                    = " ${ISA})
  message(STATUS "[cMake  ]   Cluster Cores          = " ${PE})
  message(STATUS "[cMake  ]   Fabric Controller      = " ${FC})
  message(STATUS "[cMake  ]   Number of used cores   = " ${NUM_CORES})
  message(STATUS "================================================================================")
  message(STATUS "")

  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(TargetLibraries/PULPOpen)
  target_include_directories(deeploypulp PUBLIC TargetLibraries/Generic/inc)

  add_subdirectory(DeeployTest)
  target_link_libraries(deeploylib INTERFACE deeploybasic deeploypulp)

endif()

if(platform STREQUAL GAP9)
  project(${TESTNAME} LANGUAGES C ASM)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/gap9/gap9_gvsoc.cmake)
  add_compile_options(
    -Wno-error=unknown-pragmas
  )

  add_compile_definitions(
    DEEPLOY_GAP9_PLATFORM
  )

  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(DeeployTest)
  setupos(${TESTNAME})
  add_subdirectory(TargetLibraries/GAP9)
  target_include_directories(deeploygap9 PUBLIC TargetLibraries/Generic/inc)

  target_link_libraries(deeploylib INTERFACE deeploybasic deeploygap9)

endif()

if(platform STREQUAL Snitch)

  if(TOOLCHAIN STREQUAL LLVM)
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/snitch/toolchain_llvm.cmake)
  else()
    set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/snitch/toolchain_gcc.cmake)
  endif()

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/snitch/snitch.cmake)

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/snitch/snitch_cluster/snitch_cluster.cmake)

  project(deeploy LANGUAGES C ASM)

  message(STATUS "============================= ${platform} Configuration ============================")
  message(STATUS "[cMake  ]   Number of total cores  = " ${NUM_CORES})
  message(STATUS "================================================================================")
  message(STATUS "")

  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(TargetLibraries/Snitch)
  target_include_directories(deeploysnitch PUBLIC TargetLibraries/Generic/inc)

  add_subdirectory(DeeployTest)
  target_link_libraries(deeploylib INTERFACE deeploybasic deeploysnitch)

endif()

if(platform STREQUAL Chimera)

  # JUNGVI: Currently only LLVM is supported by the Chimera SDK
  set(CMAKE_TOOLCHAIN_FILE ${CMAKE_CURRENT_LIST_DIR}/cmake/chimera/toolchain_llvm.cmake)

  include(${CMAKE_CURRENT_LIST_DIR}/cmake/chimera/chimera-sdk.cmake)

  project(deeploy LANGUAGES C ASM)

  set(TARGET_PLATFORM "chimera-open")
  set(TEST_MODE "none")

  message(STATUS "============================= ${platform} Configuration ============================")
  message(STATUS "[cMake  ]   TARGET_PLATFORM          = " ${TARGET_PLATFORM})
  message(STATUS "[cMake  ]   ISA                      = " ${ISA})
  message(STATUS "[cMake  ]   TEST_MODE                = " ${TEST_MODE})
  message(STATUS "====================================================================================")
  message(STATUS "")

  add_subdirectory(TargetLibraries/Generic)
  add_subdirectory(TargetLibraries/Chimera)
  target_include_directories(deeploychimera PUBLIC TargetLibraries/Generic/inc)

  add_subdirectory(DeeployTest)
  target_link_libraries(deeploylib INTERFACE deeploybasic deeploychimera)

endif()


print_simulation_config()
