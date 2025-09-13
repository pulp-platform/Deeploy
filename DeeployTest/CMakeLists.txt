# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

include_directories(${GENERATED_SOURCE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(network OBJECT ${GENERATED_SOURCE}/Network.c)
target_link_libraries(network PUBLIC deeploylib)

if(platform STREQUAL MemPool)
  add_subdirectory(Platforms/MemPool)

elseif(platform STREQUAL SoftHier)
  add_subdirectory(Platforms/SoftHier)

elseif(platform STREQUAL Generic)
  add_subdirectory(Platforms/Generic)

elseif(DEEPLOY_ARCH STREQUAL CMSIS)
  if(platform STREQUAL QEMU-ARM)
    add_subdirectory(Platforms/QEMU_ARM)
  endif()

elseif(DEEPLOY_ARCH STREQUAL PULP)
  file(GLOB_RECURSE HEXLIST
    "${GENERATED_SOURCE}/hex/**"
  )

  if (NOT HEXLIST)
    target_compile_options(network PUBLIC -DNOFLASH)
  else()
    gvsoc_flags_add_files_to_hyperflash(GVSOC_HEX_HYPERFLASH_FLAGS HEXLIST)
    list(APPEND GVSOC_EXTRA_FLAGS ${GVSOC_HEX_HYPERFLASH_FLAGS})
  endif()

  # SCHEREMO: Waive warnings
  # Pointer sign warnings are caused by the data width abstraction used in Deeploy. Signedness is not explicitly modelled, as this is handled by kernels
  target_compile_options(network PRIVATE
    -Wno-pointer-sign
  )

  if(platform STREQUAL Siracusa OR platform STREQUAL Siracusa_w_neureka)
    add_subdirectory(Platforms/Siracusa)
  elseif(platform STREQUAL PULPOpen)
    add_subdirectory(Platforms/PULPOpen)
  endif()
elseif(DEEPLOY_ARCH STREQUAL SNITCH)
  add_subdirectory(Platforms/Snitch)
elseif(DEEPLOY_ARCH STREQUAL CHIMERA)
  add_subdirectory(Platforms/Chimera)

endif()
