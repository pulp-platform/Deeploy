# Copyright (C) 2026 EPFL.
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# File: CMakeLists.txt
# Author: Mohammad Hossein Nikkhah
# Description: 

if(NOT DEFINED XHEEP_CRT_SOURCES OR NOT DEFINED XHEEP_RUNTIME_SOURCES)
  message(FATAL_ERROR "Include cmake/xheep/xheep.cmake before TargetLibraries/xheep.")
endif()

if(NOT XHEEP_RUNTIME_SOURCES)
  message(FATAL_ERROR "XHEEP_RUNTIME_SOURCES is empty. Check XHEEP_HOME and x-heep source collection.")
endif()

set_source_files_properties(${XHEEP_CRT_SOURCES} PROPERTIES COMPILE_FLAGS -DLANGUAGE_ASSEMBLY)

add_library(xheep-runtime OBJECT
  ${XHEEP_CRT_SOURCES}
  ${XHEEP_RUNTIME_SOURCES}
)

# Keep X-HEEP startup/runtime objects out of project-wide LTO.
set_property(TARGET xheep-runtime PROPERTY INTERPROCEDURAL_OPTIMIZATION FALSE)

target_include_directories(xheep-runtime SYSTEM PUBLIC
  ${XHEEP_INCLUDE_DIRS}
)

add_library(deeployxheep INTERFACE)

target_include_directories(deeployxheep INTERFACE
  ${XHEEP_INCLUDE_DIRS}
)

target_link_libraries(deeployxheep INTERFACE xheep-runtime)
target_sources(deeployxheep INTERFACE $<TARGET_OBJECTS:xheep-runtime>)
