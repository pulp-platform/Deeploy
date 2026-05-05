# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0


macro(add_board_deployment name target)

    if(NOT DEFINED GVSOC_INSTALL_DIR)
        message(FATAL_ERROR "Environment variable GVSOC_INSTALL_DIR not set")
    endif()

    message(STATUS "[Deeploy GAP9] Running on Board")

    set(BOARD_WORKDIR ${CMAKE_BINARY_DIR}/board_workdir)
    set(DEEPLOY_BINARY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
    set(GAP9_SDK_HOME $ENV{GAP_SDK_HOME})
    set(GAPY "${GAP9_SDK_HOME}/utils/gapy_v2/bin/gapy")
    set(FLASH_LAYOUT "${GAP9_SDK_HOME}/utils/layouts/default_layout_multi_readfs.json")
    set(FSBL_BINARY "${GAP9_SDK_HOME}/install/target/bin/fsbl")
    set(SSBL_BINARY "${GAP9_SDK_HOME}/install/target/bin/ssbl")

    make_directory(${BOARD_WORKDIR})

    if(NOT DEFINED GAP9_SDK_HOME)
        message(FATAL_ERROR "Environment variable GAP_SDK_HOME not set")
    endif()

    # Command to run on board
    set(GAPY_CMD
        ${GAPY}
        --target=gap9.evk
        --platform=board
        --target-property=boot.flash_device=mram
        --target-property=boot.mode=flash
        --target-dir=${GAP9_SDK_HOME}/utils/gapy_v2/targets
        --openocd-cable=${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gapuino_ftdi.cfg
        --openocd-script=${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gap9revb.tcl
        --openocd-tools=${GAP9_SDK_HOME}/utils/openocd_tools
        --binary=${DEEPLOY_BINARY}
        --work-dir=${BOARD_WORKDIR}
        --multi-flash-content=${FLASH_LAYOUT}
        --flash-size=67108864
        --flash-property=${FSBL_BINARY}@mram:fsbl:binary
        --flash-property=${SSBL_BINARY}@mram:ssbl:binary
        --flash-property=${DEEPLOY_BINARY}@mram:app:binary run
        --py-stack
    )

    # Add readfs files if provided
    if(GAPY_RUNNER_ARGS)
        list(LENGTH GAPY_RUNNER_ARGS num_readfs_files)
        message(STATUS "[Deeploy GAP9] Adding ${num_readfs_files} readfs file(s)")
        list(APPEND GAPY_CMD ${GAPY_RUNNER_ARGS})
    endif()

    # Convert list to string for printing
    string(REPLACE ";" " " GAPY_CMD_STR "${GAPY_CMD}")

    add_custom_target(board_${name}
        DEPENDS ${name}
        WORKING_DIRECTORY ${BOARD_WORKDIR}
        COMMAND bash -c "${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/*.bin ${BOARD_WORKDIR}/ 2>/dev/null || true"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GAP9_SDK_HOME}/utils/efuse/GAP9/efuse_hyper_preload.data
            ${BOARD_WORKDIR}/chip.efuse_preload.data
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${CMAKE_COMMAND} -E echo "[Deeploy GAP9] Executing gapy command to run on board:"
        COMMAND ${CMAKE_COMMAND} -E echo "${GAPY_CMD_STR}"
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${GAPY_CMD}
        COMMENT "Running ${name} with gapy on GAP9 board"
        POST_BUILD
        USES_TERMINAL
        VERBATIM
    )
endmacro()