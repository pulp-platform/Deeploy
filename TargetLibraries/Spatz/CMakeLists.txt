file(GLOB_RECURSE SOURCES
	"src/**"
)

include(cmake/spatz-runtime-precompiled.cmake)

add_deeploy_library(deeployspatz STATIC ${SOURCES})
target_include_directories(deeployspatz
	PUBLIC
	${CMAKE_CURRENT_LIST_DIR}/inc
)
target_include_directories(deeployspatz SYSTEM PUBLIC ${SPATZ_RUNTIME_INCLUDE})
target_link_libraries(deeployspatz INTERFACE spatz-runtime)
