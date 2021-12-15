include_directories(${GENERATED_SOURCE})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(network OBJECT ${GENERATED_SOURCE}/Network.c)
target_link_libraries(network PUBLIC deeploylib)

if(platform STREQUAL MemPool)
  add_subdirectory(Platforms/MemPool)

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
  list(TRANSFORM HEXLIST PREPEND "--config-opt=flash/content/partitions/readfs/files=")
  set(GVSOCHEXINCLUDE ${HEXLIST})

  if (NOT HEXLIST)
    target_compile_options(network PUBLIC
      -DNOFLASH
    )
  endif()
  # SCHEREMO: Waive warnings
  # Pointer sign warnings are caused by the data width abstraction used in Deeploy. Signedness is not explicitly modelled, as this is handled by kernels
  target_compile_options(network PRIVATE
    -Wno-pointer-sign
  )

  if(platform STREQUAL Siracusa OR platform STREQUAL Siracusa_w_neureka)
    set(USE_NEUREKA ON)
    add_subdirectory(Platforms/Siracusa)
  elseif(platform STREQUAL PULPOpen)
    set(USE_NEUREKA OFF)
    add_subdirectory(Platforms/PULPOpen)
  endif()
endif()
