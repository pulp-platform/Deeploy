set(SNITCH_HOME $ENV{SNITCH_HOME})
set(SNITCH_RUNTIME_HOME ${SNITCH_HOME}/sw/snRuntime)

add_compile_definitions(
  DEEPLOY_SNITCH_PLATFORM
)

set(DEEPLOY_ARCH SNITCH)

set(num_threads  ${NUM_CORES})

macro(add_snitch_cluster_vsim_simulation name)
    add_custom_target(vsim_${name}
	WORKING_DIRECTORY ${SNITCH_HOME}/target/snitch_cluster
	DEPENDS ${name}
	COMMAND ${QUESTA} bin/snitch_cluster.vsim
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name} || true
	COMMENT "Simulating deeploytest with vsim"
	POST_BUILD
	USES_TERMINAL
	VERBATIM
    )
endmacro()

macro(add_snitch_cluster_vsim_gui_simulation name)
    add_custom_target(vsim.gui_${name}
	WORKING_DIRECTORY ${SNITCH_HOME}/target/snitch_cluster
	DEPENDS ${name}
	COMMAND ${QUESTA} bin/snitch_cluster.vsim.gui
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name} || true
	COMMENT "Simulating deeploytest with vsim.gui"
	POST_BUILD
	USES_TERMINAL
	VERBATIM
    )
endmacro()