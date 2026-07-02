add_compile_definitions(
	DEEPLOY_SPATZ_PLATFORM
)

set(DEEPLOY_ARCH SPATZ)

set(num_threads ${NUM_CORES})

macro(add_spatz_vsim_simulation name)
	add_custom_target(vsim_${name}
	WORKING_DIRECTORY ${SPATZ_HOME}/hw/system/spatz_cluster
	DEPENDS ${name}
	COMMAND ${QUESTA} bin/spatz_cluster.vsim
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name} || true
	COMMENT "Simulating deeploytest with vsim (Spatz cluster)"
	POST_BUILD
	USES_TERMINAL
	VERBATIM
	)
endmacro()

add_compile_options(
		-ffast-math
)

add_link_options(
		-ffast-math
		-Wl,--gc-sections
)

