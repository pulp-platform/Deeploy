add_compile_definitions(
  DEEPLOY_PULP_PLATFORM
)

set(DEEPLOY_ARCH PULP)

macro(add_gvsoc_emulation name)
  add_custom_target(gvsoc_${name}
    DEPENDS ${name}
    COMMAND $ENV{GVSOC_INSTALL_DIR}/bin/gvsoc --target siracusa --binary ${CMAKE_BINARY_DIR}/bin/${name} run
    COMMENT "Simulating deeploytest with GVSOC"
    POST_BUILD
    USES_TERMINAL
    VERBATIM
  )
endmacro()
