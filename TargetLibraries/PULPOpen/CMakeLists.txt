file(GLOB_RECURSE SOURCES
  "src/**"
)

if(NOT DEFINED ENV{PULP_SDK_HOME})
  message(FATAL_ERROR "Environment variable PULP_SDK_HOME not set.")
endif()

if(platform STREQUAL "Siracusa" OR platform STREQUAL "Siracusa_w_neureka")
  include(cmake/pulp-sdk-siracusa.cmake)
elseif(platform STREQUAL "PULPOpen")
  include(cmake/pulp-sdk-pulp-open.cmake)
else()
  message(FATAL_ERROR "Unrecognized platform: ${platform}")
endif()

add_deeploy_library(deeploypulp STATIC ${SOURCES})
target_include_directories(deeploypulp
  PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/inc
)

target_compile_options(deeploypulp PRIVATE
  -Wno-implicit-function-declaration
  -Wno-implicit-int-conversion
  -Wno-sign-conversion
  -Wno-sign-compare
)

target_include_directories(deeploypulp PUBLIC ${PULP_SDK_INCLUDES})
target_compile_options(deeploypulp PUBLIC ${PULP_SDK_COMPILE_FLAGS})

add_subdirectory(third_party/pulp-nn-mixed)
target_include_directories(pulp-nn-mixed PUBLIC ${PULP_SDK_INCLUDES})
target_compile_options(pulp-nn-mixed PUBLIC ${PULP_SDK_COMPILE_FLAGS})

target_link_libraries(deeploypulp PUBLIC pulp-nn-mixed)
target_link_libraries(deeploypulp INTERFACE pulp-sdk)
target_sources(deeploypulp INTERFACE $<TARGET_OBJECTS:pulp-sdk>)

set(PULP_NNX_PLATFORMS "Siracusa;Siracusa_w_neureka")
if (platform IN_LIST PULP_NNX_PLATFORMS)
  if (platform STREQUAL "Siracusa" OR platform STREQUAL "Siracusa_w_neureka")
    set(USE_NEUREKA ON)
  else()
    message(FATAL_ERROR "Missing accelerator flags for platform ${platform}")
  endif()
  add_subdirectory(third_party/pulp-nnx)
  target_include_directories(pulp-nnx PUBLIC ${PULP_SDK_INCLUDES})
  target_compile_options(pulp-nnx PUBLIC ${PULP_SDK_COMPILE_FLAGS})
  target_link_libraries(deeploypulp INTERFACE pulp-nnx)
endif()
