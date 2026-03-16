file(GLOB_RECURSE SOURCES
	"src/**"
)

include(cmake/snitch-runtime-precompiled.cmake)

add_deeploy_library(deeployspatz STATIC ${SOURCES})
target_include_directories(deeployspatz
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}/inc
)
target_include_directories(deeployspatz SYSTEM PUBLIC ${SNITCH_RUNTIME_INCLUDE})
target_compile_options(deeployspatz PUBLIC ${SNITCH_RUNTIME_COMPILE_FLAGS})
target_link_libraries(deeployspatz INTERFACE snitch-runtime)
