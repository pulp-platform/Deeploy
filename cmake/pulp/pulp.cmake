# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

set(PULP_OPEN_HOME $ENV{PULP_OPEN_HOME})
set(PULP_OPEN_RUNTIME_HOME $ENV{PULP_SDK_HOME})

set(SVLIB $ENV{PULP_OPEN_HOME}/rtl/tb/remote_bitbang/librbs)
set(VSIM_FLAGS -gUSE_SDVT_SPI=0 -gUSE_SDVT_CPI=0 -gBAUDRATE=115200 -gENABLE_DEV_DPI=0 -gLOAD_L2=JTAG -gUSE_SDVT_I2S=0)

add_compile_definitions(
  DEEPLOY_PULP_PLATFORM
)

set(DEEPLOY_ARCH PULP)

set(num_threads  ${NUM_CORES})

macro(add_pulp_open_qsim_simulation name)

  set(TARGET_NAME ${name})
  set(TARGETS "${CMAKE_BINARY_DIR}/bin/${TARGET_NAME}")

  set(TARGET_BUILD_DIR ${PULP_OPEN_HOME}/sim)
  set(TARGET_BIN_DIR ${CMAKE_BINARY_DIR}/)
  set(VSIM_PATH ${PULP_OPEN_HOME}/sim)

  message(STATUS "PULP_OPEN_HOME = ${PULP_OPEN_HOME}")
  message(STATUS "TARGET_BUILD_DIR = ${TARGET_BUILD_DIR}")

  set(SIM_DIRS
    ${TARGET_BUILD_DIR}/stdout
    ${TARGET_BUILD_DIR}/fs
    ${TARGET_BUILD_DIR}/waves
    ${TARGET_BUILD_DIR}/boot
    ${TARGET_BUILD_DIR}/tcl_files
  )

  foreach(dir ${SIM_DIRS})
    add_custom_command(
      OUTPUT ${dir}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${dir}
      COMMENT "Creating ${dir}"
    )
  endforeach()

  set(ENTRY_POINT "0x1c008080")

  add_custom_target(qsim.gui_${name}
    DEPENDS ${name} ${SIM_DIRS}
    WORKING_DIRECTORY ${TARGET_BUILD_DIR}

    COMMAND ${CMAKE_COMMAND} -E env
      VSIM_PATH=${VSIM_PATH}
      VSIM_RUNNER_FLAGS=+ENTRY_POINT=${ENTRY_POINT}
      $ENV{PULP_SDK_HOME}/bin/stim_utils.py --binary=${TARGETS} --vectors=${TARGET_BUILD_DIR}/vectors/stim.txt

    COMMAND $ENV{PULP_SDK_HOME}/bin/plp_mkflash
      --flash-boot-binary=${TARGETS}
      --stimuli=${TARGET_BUILD_DIR}/vectors/qspi_stim.slm
      --flash-type=spi --qpi

    COMMAND $ENV{PULP_SDK_HOME}/bin/slm_hyper.py
      --input=${TARGET_BUILD_DIR}/vectors/qspi_stim.slm
      --output=${TARGET_BUILD_DIR}/vectors/hyper_stim.slm

    COMMAND ${CMAKE_COMMAND} -E env USE_QONE=1
      VSIM_RUNNER_FLAGS=+ENTRY_POINT=${ENTRY_POINT}
      qsim -do "source ${VSIM_PATH}/tcl_files/config/run_and_exit.tcl"
        -do "source ${VSIM_PATH}/tcl_files/run.tcl; "

    COMMENT "Simulating ${name} with qsim"
    POST_BUILD
    USES_TERMINAL
    VERBATIM
  )

  add_custom_target(qsim_${name}
    DEPENDS ${name} ${SIM_DIRS}
    WORKING_DIRECTORY ${TARGET_BUILD_DIR}

    COMMAND ${CMAKE_COMMAND} -E env
      VSIM_PATH=${VSIM_PATH}
      VSIM_RUNNER_FLAGS=+ENTRY_POINT=${ENTRY_POINT}
      $ENV{PULP_SDK_HOME}/bin/stim_utils.py --binary=${TARGETS} --vectors=${TARGET_BUILD_DIR}/vectors/stim.txt

    COMMAND $ENV{PULP_SDK_HOME}/bin/plp_mkflash
      --flash-boot-binary=${TARGETS}
      --stimuli=${TARGET_BUILD_DIR}/vectors/qspi_stim.slm
      --flash-type=spi --qpi

    COMMAND $ENV{PULP_SDK_HOME}/bin/slm_hyper.py
      --input=${TARGET_BUILD_DIR}/vectors/qspi_stim.slm
      --output=${TARGET_BUILD_DIR}/vectors/hyper_stim.slm

    COMMAND ${CMAKE_COMMAND} -E env USE_QONE=1
      VSIM_RUNNER_FLAGS=+ENTRY_POINT=${ENTRY_POINT}
      qsim -c -do "source ${VSIM_PATH}/tcl_files/config/run_and_exit.tcl"
        -do "source ${VSIM_PATH}/tcl_files/run.tcl;"

    COMMENT "Simulating ${name} with qsim"
    POST_BUILD
    USES_TERMINAL
    VERBATIM
  )
endmacro()

macro(add_pulp_open_vsim_simulation name)

  set(TARGET_NAME ${name})
  set(TARGETS "${CMAKE_BINARY_DIR}/bin/${TARGET_NAME}")

  set(TARGET_BUILD_DIR ${PULP_OPEN_HOME}/sim)
  set(TARGET_BIN_DIR ${CMAKE_BINARY_DIR}/)
  set(VSIM_PATH ${PULP_OPEN_HOME}/sim)

  message(STATUS "PULP_OPEN_HOME = ${PULP_OPEN_HOME}")
  message(STATUS "TARGET_BUILD_DIR = ${TARGET_BUILD_DIR}")

  set(SIM_DIRS
    ${TARGET_BUILD_DIR}/stdout
    ${TARGET_BUILD_DIR}/fs
    ${TARGET_BUILD_DIR}/waves
    ${TARGET_BUILD_DIR}/boot
    ${TARGET_BUILD_DIR}/tcl_files
  )

  foreach(dir ${SIM_DIRS})
    add_custom_command(
      OUTPUT ${dir}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${dir}
      COMMENT "Creating ${dir}"
    )
  endforeach()

  set(ENTRY_POINT "0x1c008080")

  add_custom_target(vsim_${name}
    DEPENDS ${name} ${SIM_DIRS}
    WORKING_DIRECTORY ${TARGET_BUILD_DIR}

    COMMAND ${CMAKE_COMMAND} -E env
      VSIM_PATH=${VSIM_PATH}
      VSIM_RUNNER_FLAGS=+ENTRY_POINT=${ENTRY_POINT}
      $ENV{PULP_SDK_HOME}/bin/stim_utils.py --binary=${TARGETS} --vectors=${TARGET_BUILD_DIR}/vectors/stim.txt

    COMMAND $ENV{PULP_SDK_HOME}/bin/plp_mkflash
      --flash-boot-binary=${TARGETS}
      --stimuli=${TARGET_BUILD_DIR}/vectors/qspi_stim.slm
      --flash-type=spi --qpi

    COMMAND $ENV{PULP_SDK_HOME}/bin/slm_hyper.py
      --input=${TARGET_BUILD_DIR}/vectors/qspi_stim.slm
      --output=${TARGET_BUILD_DIR}/vectors/hyper_stim.slm

    COMMAND ${CMAKE_COMMAND} -E env
      VSIM_RUNNER_FLAGS=+ENTRY_POINT=${ENTRY_POINT}
      ${QUESTA} -64
        -gBAUDRATE=115200
        -gLOAD_L2=JTAG
        -permit_unmatched_virtual_intf
        -do "source ${VSIM_PATH}/tcl_files/config/run_and_exit.tcl"
        -do "source ${VSIM_PATH}/tcl_files/run.tcl;"

    COMMENT "Simulating ${name} with vsim"
    POST_BUILD
    USES_TERMINAL
    VERBATIM
  )
endmacro()


add_compile_options(
  -ffast-math
)

add_link_options(
  -ffast-math
  -Wl,--gc-sections
)
