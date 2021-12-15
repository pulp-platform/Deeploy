set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
set(CMAKE_EXPORT_COMPILE_COMMANDS TRUE)

set(use_dma 1 CACHE STRING "Enable DMA trasfers")

add_compile_definitions(
    USE_DMA=${use_dma}
)

add_library(deeploylib INTERFACE)

add_compile_options(
    -std=gnu99

    -ffast-math
    -fdiagnostics-color=always

    -Wunused-variable
    -Wconversion
    -Wall
    -Wextra

    -O2
    -g
    -ffunction-sections
    -fdata-sections
)

add_link_options(
    -std=gnu99

    -ffast-math
    -fdiagnostics-color=always

    -Wunused-variable
    -Wconversion
    -Wall
    -Wextra
    -Wl,--gc-sections
)
