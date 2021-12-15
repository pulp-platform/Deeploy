add_compile_definitions(
  DEEPLOY_PULP_PLATFORM
)

set(DEEPLOY_ARCH PULP)

macro(add_gvsoc_emulation name)
  add_custom_target(gvsoc_${name}
    DEPENDS ${name}
    COMMAND gapy --target=siracusa --platform=gvsoc --work-dir=${CMAKE_BINARY_DIR}/bin --config-opt=cluster/nb_pe=8  ${GVSOCHEXINCLUDE} --config-opt=**/runner/verbose=true -v run --image --binary=${CMAKE_BINARY_DIR}/bin/${name} > /dev/null
    COMMAND gapy --target=siracusa --platform=gvsoc --work-dir=${CMAKE_BINARY_DIR}/bin --config-opt=cluster/nb_pe=8  ${GVSOCHEXINCLUDE} --config-opt=**/runner/verbose=true -v run --flash --binary=${CMAKE_BINARY_DIR}/bin/${name} > /dev/null
    COMMAND gapy --target=siracusa --platform=gvsoc --work-dir=${CMAKE_BINARY_DIR}/bin --config-opt=cluster/nb_pe=8  ${GVSOCHEXINCLUDE} --config-opt=**/runner/verbose=true -v run --exec-prepare --exec --binary=${CMAKE_BINARY_DIR}/bin/${name}
    COMMENT "Simulating deeploytest with GVSOC"
    POST_BUILD
    USES_TERMINAL
    VERBATIM
  )
endmacro()
