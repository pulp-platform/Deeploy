# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

# Keep compatibility with existing runtime cmake snippets expecting SNITCH_* vars.

add_compile_definitions(
	DEEPLOY_SPATZ_PLATFORM
)

set(DEEPLOY_ARCH SPATZ)

set(num_threads ${NUM_CORES})

# TODO check this
# macro(add_spatz_vsim_simulation name)
# 		if(EXISTS ${SPATZ_HOME}/target/spatz_cluster/bin/spatz_cluster.vsim)
# 			set(_SPATZ_VSIM_WORKDIR ${SPATZ_HOME}/target/spatz_cluster)
# 			set(_SPATZ_VSIM_BIN bin/spatz_cluster.vsim)
# 		else()
# 			set(_SPATZ_VSIM_WORKDIR ${SPATZ_HOME}/target/snitch_cluster)
# 			set(_SPATZ_VSIM_BIN bin/snitch_cluster.vsim)
# 		endif()
# 
# 		add_custom_target(vsim_${name}
# 	WORKING_DIRECTORY ${_SPATZ_VSIM_WORKDIR}
# 	DEPENDS ${name}
# 	COMMAND ${QUESTA} ${_SPATZ_VSIM_BIN}
# 	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name} || true
# 	COMMENT "Simulating deeploytest with vsim (Spatz)"
# 	POST_BUILD
# 	USES_TERMINAL
# 	VERBATIM
# 		)
# endmacro()

# macro(add_spatz_vsim_gui_simulation name)
# 		if(EXISTS ${SPATZ_HOME}/target/spatz_cluster/bin/spatz_cluster.vsim.gui)
# 			set(_SPATZ_VSIM_GUI_WORKDIR ${SPATZ_HOME}/target/spatz_cluster)
# 			set(_SPATZ_VSIM_GUI_BIN bin/spatz_cluster.vsim.gui)
# 		else()
# 			set(_SPATZ_VSIM_GUI_WORKDIR ${SPATZ_HOME}/target/snitch_cluster)
# 			set(_SPATZ_VSIM_GUI_BIN bin/snitch_cluster.vsim.gui)
# 		endif()
# 
# 		add_custom_target(vsim.gui_${name}
# 	WORKING_DIRECTORY ${_SPATZ_VSIM_GUI_WORKDIR}
# 	DEPENDS ${name}
# 	COMMAND ${QUESTA} ${_SPATZ_VSIM_GUI_BIN}
# 	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name} || true
# 	COMMENT "Simulating deeploytest with vsim.gui (Spatz)"
# 	POST_BUILD
# 	USES_TERMINAL
# 	VERBATIM
# 		)
# endmacro()

add_compile_options(
		-ffast-math
)

add_link_options(
		-ffast-math
		-Wl,--gc-sections
)

