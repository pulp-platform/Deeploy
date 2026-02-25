# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

include_directories(${GENERATED_SOURCE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(TRAINING)
  add_library(training_network OBJECT ${GENERATED_SOURCE}/TrainingNetwork.c)
  target_link_libraries(training_network PUBLIC deeploylib)
  # Optimizer network (SGD kernel, compiled separately to allow different prefix)
  add_library(optimizer_network OBJECT ${GENERATED_SOURCE}/OptimizerNetwork.c)
  target_link_libraries(optimizer_network PUBLIC deeploylib)
else()
  add_library(network OBJECT ${GENERATED_SOURCE}/Network.c)
  target_link_libraries(network PUBLIC deeploylib)
endif()

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
    if(TRAINING)
      target_compile_options(training_network PUBLIC -DNOFLASH)
      target_compile_options(optimizer_network PUBLIC -DNOFLASH)
    else()
      target_compile_options(network PUBLIC -DNOFLASH)
    endif()
  else()
    gvsoc_flags_add_files_to_hyperflash(GVSOC_HEX_HYPERFLASH_FLAGS HEXLIST)
    list(APPEND GVSOC_EXTRA_FLAGS ${GVSOC_HEX_HYPERFLASH_FLAGS})
  endif()

  # SCHEREMO: Waive warnings
  # Pointer sign warnings are caused by the data width abstraction used in Deeploy. Signedness is not explicitly modelled, as this is handled by kernels
  if(TRAINING)
    target_compile_options(training_network PRIVATE -Wno-pointer-sign)
    target_compile_options(optimizer_network PRIVATE -Wno-pointer-sign)
  else()
    target_compile_options(network PRIVATE -Wno-pointer-sign)
  endif()

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
