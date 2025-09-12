# Copyright 2025 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# SoftHier runtime should have structure:
# runtime/
# ├── include/
# │   └── flex_*.h     -- runtime functions
# ├── flex_memory.ld   -- linker script
# └── flex_start.s     -- customised asm script

set(SOFTHIER_SDK_HOME $ENV{SOFTHIER_INSTALL_DIR}/soft_hier/flex_cluster_sdk)
set(SOFTHIER_RUNTIME_HOME ${SOFTHIER_SDK_HOME}/runtime)


# runtime libraries from SoftHier directory
set(SOFTHIER_INCLUDES
    ${SOFTHIER_RUNTIME_HOME}/include
)

set(SOFTHIER_RUNTIME_ASM_SOURCE
    ${SOFTHIER_RUNTIME_HOME}/flex_start.s
)
set_source_files_properties(
    ${SOFTHIER_RUNTIME_ASM_SOURCE} PROPERTIES COMPILE_FLAGS -DLANGUAGE_ASSEMBLY
)



add_library(softhier-sdk OBJECT
            ${SOFTHIER_RUNTIME_ASM_SOURCE}
)

target_compile_options(softhier-sdk PRIVATE
  -I${SOFTHIER_RUNTIME_HOME}/include
)
target_include_directories(softhier-sdk SYSTEM PUBLIC ${SOFTHIER_INCLUDES})

target_compile_options(softhier-sdk PRIVATE
    -Wno-sign-conversion
    -Wno-unused-function
    -Wno-unused-parameter
    -Wno-conversion
    -Wno-sign-conversion
    -Wno-unused-variable
    -Wno-sign-compare
    -Wno-return-type
    -fno-inline-functions
    -fno-strict-aliasing
    -Wimplicit-function-declaration
)

target_compile_options(softhier-sdk INTERFACE
  -Wno-unused-function
  -Wno-int-conversion
  -Wimplicit-function-declaration
)

set(SOFTHIER_LINKER_SCRIPT
${SOFTHIER_RUNTIME_HOME}/flex_memory.ld
)

set(SOFTHIER_LINK_OPTIONS
  -Wl,--gc-sections
  -T${SOFTHIER_LINKER_SCRIPT}
)

target_link_libraries(softhier-sdk PUBLIC
  ${SOFTHIER_LINK_OPTIONS}
)