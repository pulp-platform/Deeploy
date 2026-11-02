# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

macro(add_deeploy_library name)
    add_library(${ARGV})
    add_custom_command(
        TARGET ${name}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS $<TARGET_FILE:${name}> > $<TARGET_FILE:${name}>.s)
endmacro()

macro(add_deeploy_executable name)
    add_executable(${ARGV})
    add_custom_command(
        TARGET ${name}
        POST_BUILD
        COMMAND ${CMAKE_OBJDUMP} -dhS $<TARGET_FILE:${name}> > $<TARGET_FILE:${name}>.s)
    if(DEEPLOY_ARCH STREQUAL XHEEP)
        if(XHEEP_LINKER STREQUAL flash_load OR XHEEP_LINKER STREQUAL flash_exec)
            add_custom_command(
                TARGET ${name}
                POST_BUILD
                COMMAND ${CMAKE_OBJCOPY} -O verilog --adjust-vma=-0x40000000 $<TARGET_FILE:${name}> $<TARGET_FILE_DIR:${name}>/${name}.hex)
        else()
            add_custom_command(
                TARGET ${name}
                POST_BUILD
                COMMAND ${CMAKE_OBJCOPY} -O verilog $<TARGET_FILE:${name}> $<TARGET_FILE_DIR:${name}>/${name}.hex)
        endif()
    endif()
endmacro()

macro(link_compile_dump name)
    add_custom_command(
        TARGET ${name}
        POST_BUILD
        COMMAND
            mkdir -p ${CMAKE_SOURCE_DIR}/DeeployTest/TEST_RECENT &&
            ln -sfn ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/DeeployTest/TEST_RECENT/build &&
            ln -sfn ${GENERATED_SOURCE} ${CMAKE_SOURCE_DIR}/DeeployTest/TEST_RECENT/src
            )
endmacro()

function(math_shell expr output)
    execute_process(COMMAND awk "BEGIN {printf ${expr}}" OUTPUT_VARIABLE __output)
    set(${output} ${__output} PARENT_SCOPE)
endfunction()
