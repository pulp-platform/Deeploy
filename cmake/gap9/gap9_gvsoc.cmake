# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Mark that GAP9-specific gvsoc emulation is defined
set(GAP9_GVSOC_DEFINED TRUE)

macro(add_gvsoc_emulation name target)
    if(NOT DEFINED GVSOC_INSTALL_DIR)
        message(FATAL_ERROR "Environment variable GVSOC_INSTALL_DIR not set")
    endif()

    set(GVSOC_WORKDIR ${CMAKE_BINARY_DIR}/gvsoc_workdir)
    make_directory(${GVSOC_WORKDIR})
    set(GVSOC_BINARY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")

    # GAP9 SDK paths
    set(GAP9_SDK_HOME $ENV{GAP_SDK_HOME})
    if(NOT DEFINED GAP9_SDK_HOME)
        message(FATAL_ERROR "Environment variable GAP_SDK_HOME not set")
    endif()

    # Check if GAPY_RUNNER_ARGS is defined and non-empty (indicates L3 with readfs files)
    if(GAPY_RUNNER_ARGS)
        # L3 mode: Use gapy with flash layout and readfs
        message(STATUS "[Deeploy GAP9] L3 mode: using gapy with readfs")
        
        set(GAPY "${GAP9_SDK_HOME}/utils/gapy_v2/bin/gapy")
        set(FLASH_LAYOUT "${GAP9_SDK_HOME}/utils/layouts/default_layout_multi_readfs.json")
        set(FSBL_BINARY "${GAP9_SDK_HOME}/install/target/bin/fsbl")
        set(SSBL_BINARY "${GAP9_SDK_HOME}/install/target/bin/ssbl")

        # Build the gapy command
        set(GAPY_CMD 
            ${GAPY}
            --target=gap9.evk
            --target-dir=${GAP9_SDK_HOME}/install/workstation/generators
            --model-dir=${GAP9_SDK_HOME}/install/workstation/models
            --platform=gvsoc
            --work-dir=${GVSOC_WORKDIR}
            --target-property=boot.flash_device=mram
            --target-property=boot.mode=flash
            --multi-flash-content=${FLASH_LAYOUT}
            --flash-property=${GVSOC_BINARY}@mram:app:binary
        )

        # Add readfs files if provided
        if(GAPY_RUNNER_ARGS)
            list(LENGTH GAPY_RUNNER_ARGS num_readfs_files)
            message(STATUS "[Deeploy GAP9] Adding ${num_readfs_files} readfs file(s)")
            list(APPEND GAPY_CMD ${GAPY_RUNNER_ARGS})
        endif()

        # Add fsbl/ssbl
        list(APPEND GAPY_CMD
            --flash-property=${FSBL_BINARY}@mram:fsbl:binary
            --flash-property=${SSBL_BINARY}@mram:ssbl:binary
        )

        # Add final commands
        list(APPEND GAPY_CMD
            --py-stack
            image flash run
            --binary=${GVSOC_BINARY}
        )

        # Convert list to string for printing
        string(REPLACE ";" " " GAPY_CMD_STR "${GAPY_CMD}")
        
        add_custom_target(gvsoc_${name}
            DEPENDS ${name}
            WORKING_DIRECTORY ${GVSOC_WORKDIR}
            COMMAND bash -c "${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/*.bin ${GVSOC_WORKDIR}/ 2>/dev/null || true"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different 
                ${GAP9_SDK_HOME}/utils/efuse/GAP9/efuse_hyper_preload.data 
                ${GVSOC_WORKDIR}/chip.efuse_preload.data
            COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
            COMMAND ${CMAKE_COMMAND} -E echo "[Deeploy GAP9] Executing gapy command (L3 mode with readfs):"
            COMMAND ${CMAKE_COMMAND} -E echo "${GAPY_CMD_STR}"
            COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
            COMMAND ${GAPY_CMD}
            COMMENT "Simulating ${name} with gapy for GAP9 (L3 mode)"
            POST_BUILD
            USES_TERMINAL
            VERBATIM
        )
        
    else()
        # L2 mode: Use traditional gvsoc command directly (no flash/readfs)
        message(STATUS "[Deeploy GAP9] L2 mode: using traditional gvsoc without flash")

        set(GVSOC_EXECUTABLE "${GVSOC_INSTALL_DIR}/bin/gvsoc")

        # L2 mode: run directly without flash operations
        set(GVSOC_CMD
            ${GVSOC_EXECUTABLE}
            --target=${target}
            --binary ${GVSOC_BINARY}
            --work-dir=${GVSOC_WORKDIR}
            image flash run
        )

        # Convert list to string for printing
        string(REPLACE ";" " " GVSOC_CMD_STR "${GVSOC_CMD}")

        add_custom_target(gvsoc_${name}
            DEPENDS ${name}
            WORKING_DIRECTORY ${GVSOC_WORKDIR}
            COMMAND ${CMAKE_COMMAND} -E copy_if_different ${CMAKE_BINARY_DIR}/*.bin ${GVSOC_WORKDIR}/ || true
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${GAP9_SDK_HOME}/utils/efuse/GAP9/efuse_hyper_preload.data
                ${GVSOC_WORKDIR}/chip.efuse_preload.data
            COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
            COMMAND ${CMAKE_COMMAND} -E echo "[Deeploy GAP9] Executing gvsoc command - L2 mode:"
            COMMAND ${CMAKE_COMMAND} -E echo "${GVSOC_CMD_STR}"
            COMMAND ${CMAKE_COMMAND} -E echo "=========================================="
            COMMAND ${GVSOC_CMD}
            COMMENT "Simulating ${name} with gvsoc for GAP9 (L2 mode)"
            POST_BUILD
            USES_TERMINAL
        )
    endif()
endmacro()