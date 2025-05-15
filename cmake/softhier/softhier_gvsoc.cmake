macro(add_gvsoc_emulation name)
  set(BINARY_PATH ${CMAKE_BINARY_DIR}/bin/${name})

  set(GVSOC_EXECUTABLE $ENV{SOFTHIER_INSTALL_DIR}/install/bin/gvsoc)

  add_custom_target(gvsoc_${name}
    DEPENDS ${name}
    COMMAND ${GVSOC_EXECUTABLE}
            --target=pulp.chips.flex_cluster.flex_cluster
            --binary ${BINARY_PATH}
            run
    COMMENT "Simulating deeploytest with GVSOC"
    USES_TERMINAL
    VERBATIM
  )
endmacro()