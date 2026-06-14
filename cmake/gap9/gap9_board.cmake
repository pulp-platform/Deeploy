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

    # Common gapy arguments shared by the flash and run phases.
    # The flash and run phases are issued as SEPARATE gapy invocations (instead
    # of a single "image flash run") so that a failed flash phase aborts the
    # build instead of silently falling through to "run" against stale MRAM.
    # A transient probe drop (e.g. "unable to open ftdi device") during the
    # flash phase otherwise lets the chip execute a previously-flashed binary,
    # which looks like a test-specific hang but is really mismatched flash.
    set(GAPY_BASE
        ${GAPY}
        --target=gap9.evk
        --platform=board
        --target-property=boot.flash_device=mram
        --target-property=boot.mode=flash
        --target-dir=${GAP9_SDK_HOME}/utils/gapy_v2/targets
        --openocd-cable=${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gapuino_ftdi.cfg
        --openocd-script=${GAP9_SDK_HOME}/utils/openocd_tools/tcl/gap9revb.tcl
        --openocd-tools=${GAP9_SDK_HOME}/utils/openocd_tools
        --work-dir=${BOARD_WORKDIR}
        --multi-flash-content=${FLASH_LAYOUT}
        --flash-size=67108864
        --flash-property=${FSBL_BINARY}@mram:fsbl:binary
        --flash-property=${SSBL_BINARY}@mram:ssbl:binary
        --flash-property=${DEEPLOY_BINARY}@mram:app:binary
    )

    # Add readfs files if provided (MUST come before the subcommand appended below)
    if(GAPY_RUNNER_ARGS)
        list(LENGTH GAPY_RUNNER_ARGS num_readfs_files)
        message(STATUS "[Deeploy GAP9] Adding ${num_readfs_files} readfs file(s)")
        list(APPEND GAPY_BASE ${GAPY_RUNNER_ARGS})
    endif()

    # Phase 1: build the flash image and write it to MRAM.
    # Subcommand + binary go LAST, after all --flash-property options.
    set(GAPY_FLASH_CMD ${GAPY_BASE} --py-stack image flash --binary=${DEEPLOY_BINARY})

    # Phase 2: execute the freshly-flashed binary. Only reached if phase 1 succeeds.
    set(GAPY_RUN_CMD ${GAPY_BASE} --py-stack run --binary=${DEEPLOY_BINARY})

    # Convert lists to strings for printing
    string(REPLACE ";" " " GAPY_FLASH_CMD_STR "${GAPY_FLASH_CMD}")
    string(REPLACE ";" " " GAPY_RUN_CMD_STR "${GAPY_RUN_CMD}")

    # CMake runs the COMMANDs below sequentially and aborts the target on the
    # first non-zero exit code, so a failed flash phase will never fall through
    # to the run phase.
    add_custom_target(board_${name}
        DEPENDS ${name}
        WORKING_DIRECTORY ${BOARD_WORKDIR}
        COMMAND bash -c "${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/*.bin ${BOARD_WORKDIR}/ 2>/dev/null || true"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${GAP9_SDK_HOME}/utils/efuse/GAP9/efuse_hyper_preload.data
            ${BOARD_WORKDIR}/chip.efuse_preload.data
        # Release a stale openocd still claiming the FT2232 probe. The previous
        # test's run-phase openocd (the load_and_start_binary bridge with its
        # telnet server) can outlive its program and keep the probe busy, which
        # makes THIS test's flash phase fail with "unable to open ftdi device"
        # and silently fall through to running mismatched flash. Terminate any
        # lingering gapuino_ftdi openocd and let the kernel release the USB
        # interface before flashing. Best-effort: never fails the build itself.
        COMMAND bash -c "pkill -f 'openocd.*gapuino_ftdi' 2>/dev/null; sleep 1; true"
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${CMAKE_COMMAND} -E echo "[Deeploy GAP9] Phase 1/2 - flashing image to MRAM:"
        COMMAND ${CMAKE_COMMAND} -E echo "${GAPY_FLASH_CMD_STR}"
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${GAPY_FLASH_CMD}
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${CMAKE_COMMAND} -E echo "[Deeploy GAP9] Phase 2/2 - running flashed binary:"
        COMMAND ${CMAKE_COMMAND} -E echo "${GAPY_RUN_CMD_STR}"
        COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
        COMMAND ${GAPY_RUN_CMD}
        COMMENT "Flashing and running ${name} with gapy on GAP9 board"
        POST_BUILD
        USES_TERMINAL
        VERBATIM
    )
endmacro()
