add_compile_definitions(
    DEEPLOY_GENERIC_PLATFORM
)

if(APPLE)
  add_link_options(
    -Wl,-dead_strip
  )
else()
  add_link_options(
    -Wl,--gc-sections
  )
endif()
