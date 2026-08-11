# Copyright (C) 2026 EPFL.
# Solderpad Hardware License, Version 2.1, see LICENSE.md for details.
# SPDX-License-Identifier: Apache-2.0 WITH SHL-2.1
#
# File: xheep_verilator.cmake
# Author: Mohammad Hossein Nikkhah
# Description: Add Deeploy CMake targets for running X-HEEP Verilator simulation.

set(XHEEP_VERILATOR_DIR "" CACHE PATH
    "Path to X-HEEP sim-verilator directory containing Vtestharness")

set(XHEEP_SIM_ARGS "" CACHE STRING
    "Extra Verilator plusargs for X-HEEP, for example '+max_sim_time=750us'")

function(add_xheep_verilator_simulation name)
  if(XHEEP_VERILATOR_DIR)
    set(_xheep_sim_dir "${XHEEP_VERILATOR_DIR}")
  else()
    file(GLOB _xheep_sim_dirs LIST_DIRECTORIES true
      "${XHEEP_HOME}/build/openhwgroup.org_systems_core-v-mini-mcu_*/sim-verilator"
    )

    list(SORT _xheep_sim_dirs)
    list(REVERSE _xheep_sim_dirs)

    if(_xheep_sim_dirs)
      list(GET _xheep_sim_dirs 0 _xheep_sim_dir)
    endif()
  endif()

  if(NOT _xheep_sim_dir)
    add_custom_target(verilator_${name}
      COMMAND ${CMAKE_COMMAND} -E echo
              "ERROR: Could not find X-HEEP sim-verilator directory under ${XHEEP_HOME}/build."
      COMMAND ${CMAKE_COMMAND} -E echo
              "Run: cd ${XHEEP_HOME} && make verilator-build"
      COMMAND ${CMAKE_COMMAND} -E false
      USES_TERMINAL
    )
    return()
  endif()

  set(_xheep_harness "${_xheep_sim_dir}/Vtestharness")

  if(NOT EXISTS "${_xheep_harness}")
    add_custom_target(verilator_${name}
      COMMAND ${CMAKE_COMMAND} -E echo
              "ERROR: Missing X-HEEP Verilator executable: ${_xheep_harness}"
      COMMAND ${CMAKE_COMMAND} -E echo
              "Run: cd ${XHEEP_HOME} && make verilator-build"
      COMMAND ${CMAKE_COMMAND} -E false
      USES_TERMINAL
    )
    return()
  endif()

  get_filename_component(_xheep_firmware_hex
    "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}.hex"
    ABSOLUTE
  )

  set(_xheep_run_args "+firmware=${_xheep_firmware_hex}")

  if(XHEEP_SIM_ARGS)
    separate_arguments(_xheep_extra_args UNIX_COMMAND "${XHEEP_SIM_ARGS}")
    list(APPEND _xheep_run_args ${_xheep_extra_args})
  endif()

  add_custom_target(verilator_${name}
    DEPENDS ${name}
    WORKING_DIRECTORY "${_xheep_sim_dir}"
    COMMAND ${CMAKE_COMMAND} -E rm -f uart0.log
    COMMAND "${_xheep_harness}" ${_xheep_run_args}
    COMMAND ${CMAKE_COMMAND} -E cat uart0.log
    COMMENT "Simulating ${name} on X-HEEP Verilator"
    USES_TERMINAL
    VERBATIM
  )
endfunction()
