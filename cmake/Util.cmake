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
endmacro()

macro(link_compile_dump name)
    add_custom_command(
        TARGET ${name}
        POST_BUILD
        COMMAND ln -sf ${CMAKE_BINARY_DIR}/compile_commands.json ${CMAKE_SOURCE_DIR}/compile_commands.json)
endmacro()

function(math_shell expr output)
    execute_process(COMMAND awk "BEGIN {printf ${expr}}" OUTPUT_VARIABLE __output)
    set(${output} ${__output} PARENT_SCOPE)
endfunction()
