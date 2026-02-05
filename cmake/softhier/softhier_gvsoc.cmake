# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

macro(add_gvsoc_emulation name)
  set(BINARY_PATH ${CMAKE_BINARY_DIR}/bin/${name})

  set(GVSOC_EXECUTABLE $ENV{SOFTHIER_INSTALL_DIR}/install/bin/gvsoc)

  add_custom_target(gvsoc_${name}
    DEPENDS ${name}
    COMMAND env LD_LIBRARY_PATH=$ENV{SOFTHIER_INSTALL_DIR}/third_party/DRAMSys:$ENV{SOFTHIER_INSTALL_DIR}/third_party/systemc_install/lib64:$ENV{LD_LIBRARY_PATH}
            ${GVSOC_EXECUTABLE}
            --target=pulp.chips.flex_cluster.flex_cluster
            --binary ${BINARY_PATH}
            run
    COMMENT "Simulating deeploytest with GVSOC"
    USES_TERMINAL
    VERBATIM
  )
endmacro()