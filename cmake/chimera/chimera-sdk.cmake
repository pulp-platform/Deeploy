add_compile_definitions(
  DEEPLOY_CHIMERA_PLATFORM
)

set(DEEPLOY_ARCH CHIMERA)

add_compile_options(
  -ffast-math
)

add_link_options(
  -ffast-math
  -Wl,--gc-sections
)


set(CHIMERA_SDK_HOME /app/Deeploy/install/chimera-sdk)

set(ABI ilp32)
set(ISA_CLUSTER_SNITCH rv32im)
set(ISA_HOST rv32imc)

