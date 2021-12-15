cmake_minimum_required(VERSION 3.12)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_C_STANDARD 99)

set(CMAKE_C_COMPILER_LAUNCHER "ccache")
set(CMAKE_CXX_COMPILER_LAUNCHER "ccache")

set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

if(TOOLCHAIN STREQUAL GCC)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

set(platform MemPool CACHE STRING "Platform (MemPool, QEMU, Siracusa, Siracusa_w_neureka, PULP-Open, Generic)")
set_property(CACHE platform PROPERTY STRINGS MemPool QEMU Siracusa Siracusa_w_neureka PULP-Open Generic)

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
elseif(platform STREQUAL Generic)
  message(STATUS "Building for platform 'Generic'")
else()
  message(FATAL_ERROR "Invalid platform '${platform}' specified!")
endif()

# Import useful functions / macros
include(${CMAKE_CURRENT_LIST_DIR}/cmake/Util.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/common.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/cmake/simulation.cmake)

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

print_simulation_config()
