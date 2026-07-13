# Copyright (C) 2026 EPFL.
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# File: xheep.cmake
# Author: Mohammad Hossein Nikkhah
# Description: 

if(NOT XHEEP_HOME AND DEFINED ENV{XHEEP_HOME})
  set(XHEEP_HOME "$ENV{XHEEP_HOME}" CACHE PATH "Path to X-HEEP checkout" FORCE)
else()
  set(XHEEP_HOME "" CACHE PATH "Path to X-HEEP checkout")
endif()

if(NOT XHEEP_HOME)
  message(FATAL_ERROR "XHEEP_HOME is not set. Pass -DXHEEP_HOME=<path-to-x-heep> or export XHEEP_HOME.")
endif()

set(XHEEP_SW_DIR "${XHEEP_HOME}/sw")
set(XHEEP_LINKER_DIR "${XHEEP_SW_DIR}/linker")
set(XHEEP_CRT_DIR "${XHEEP_SW_DIR}/device/lib/crt")
set(XHEEP_RUNTIME_DIR "${XHEEP_SW_DIR}/device/lib/runtime")
set(XHEEP_DEVICE_DIR "${XHEEP_SW_DIR}/device")
set(XHEEP_TARGET sim CACHE STRING "X-HEEP software target")
set(XHEEP_LINKER on_chip CACHE STRING "X-HEEP linker mode")
set(XHEEP_COMPILER_PREFIX "riscv32-unknown-" CACHE STRING "X-HEEP GCC compiler prefix")
set_property(CACHE XHEEP_LINKER PROPERTY STRINGS on_chip flash_load flash_exec)

if(XHEEP_LINKER STREQUAL on_chip)
  set(XHEEP_LINKER_FILE link.ld)
  set(XHEEP_CRT_TYPE ON_CHIP)
elseif(XHEEP_LINKER STREQUAL flash_load)
  set(XHEEP_LINKER_FILE link_flash_load.ld)
  set(XHEEP_CRT_TYPE FLASH_LOAD)
elseif(XHEEP_LINKER STREQUAL flash_exec)
  set(XHEEP_LINKER_FILE link_flash_exec.ld)
  set(XHEEP_CRT_TYPE FLASH_EXEC)
else()
  message(FATAL_ERROR "Unsupported XHEEP_LINKER '${XHEEP_LINKER}'. Use on_chip, flash_load, or flash_exec.")
endif()

set(XHEEP_LINKER_SCRIPT "${XHEEP_LINKER_DIR}/${XHEEP_LINKER_FILE}")
set(XHEEP_CRT_SOURCES
  "${XHEEP_CRT_DIR}/crt0.S"
  "${XHEEP_CRT_DIR}/vectors.S"
)
set(XHEEP_RUNTIME_SOURCES
  "${XHEEP_RUNTIME_DIR}/core_v_mini_mcu.c"
  "${XHEEP_RUNTIME_DIR}/handler.c"
  "${XHEEP_RUNTIME_DIR}/init.c"
  "${XHEEP_RUNTIME_DIR}/syscalls.c"
  "${XHEEP_DEVICE_DIR}/lib/base/memory.c"
  "${XHEEP_DEVICE_DIR}/lib/base/mmio.c"
  "${XHEEP_DEVICE_DIR}/lib/drivers/soc_ctrl/soc_ctrl.c"
  "${XHEEP_DEVICE_DIR}/lib/drivers/uart/uart.c"
  "${XHEEP_DEVICE_DIR}/lib/drivers/fast_intr_ctrl/fast_intr_ctrl.c"
)

set(XHEEP_REQUIRED_FILES
  "${XHEEP_LINKER_SCRIPT}"
  ${XHEEP_CRT_SOURCES}
  ${XHEEP_RUNTIME_SOURCES}
  "${XHEEP_RUNTIME_DIR}/core_v_mini_mcu.h"
  "${XHEEP_RUNTIME_DIR}/core_v_mini_mcu_memory.h"
  "${XHEEP_DEVICE_DIR}/target/${XHEEP_TARGET}/x-heep.h"
  "${XHEEP_DEVICE_DIR}/lib/drivers/fast_intr_ctrl/fast_intr_ctrl.c"
)

foreach(XHEEP_REQUIRED_FILE IN LISTS XHEEP_REQUIRED_FILES)
  if(NOT EXISTS "${XHEEP_REQUIRED_FILE}")
    message(FATAL_ERROR
      "Required X-HEEP file not found: ${XHEEP_REQUIRED_FILE}\n"
      "Run `make mcu-gen` in ${XHEEP_HOME}, then reconfigure Deeploy.")
  endif()
endforeach()

file(GLOB_RECURSE XHEEP_DEVICE_HEADERS CONFIGURE_DEPENDS "${XHEEP_DEVICE_DIR}/*.h")
set(XHEEP_INCLUDE_DIRS
  "${XHEEP_SW_DIR}"
  "${XHEEP_DEVICE_DIR}"
  "${XHEEP_DEVICE_DIR}/target/${XHEEP_TARGET}"
)
foreach(XHEEP_HEADER IN LISTS XHEEP_DEVICE_HEADERS)
  get_filename_component(XHEEP_HEADER_DIR "${XHEEP_HEADER}" DIRECTORY)
  list(APPEND XHEEP_INCLUDE_DIRS "${XHEEP_HEADER_DIR}")
endforeach()
list(REMOVE_DUPLICATES XHEEP_INCLUDE_DIRS)
include_directories(SYSTEM ${XHEEP_INCLUDE_DIRS})

set(DEEPLOY_ARCH XHEEP)

add_compile_definitions(
  DEEPLOY_XHEEP_PLATFORM
  DEEPLOY_GENERIC_PLATFORM
  HOST_BUILD
  ${XHEEP_CRT_TYPE}
  INTERNAL_CRTO
  portasmHANDLE_INTERRUPT=vSystemIrqHandler
)

set(XHEEP_GCC_LIB_DIR "${TOOLCHAIN_INSTALL_DIR}/${XHEEP_COMPILER_PREFIX}elf/lib")
add_link_options(
  -T "${XHEEP_LINKER_SCRIPT}"
  -static
  -Wl,--gc-sections
  "-L${XHEEP_GCC_LIB_DIR}"
  -specs=nano.specs
)
