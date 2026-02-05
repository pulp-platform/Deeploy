# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

#########################
##  Simulation Config  ##
#########################

set(QUESTA questa-2022.3 CACHE STRING "QuestaSim version for RTL simulation")
set(VERILATOR verilator-4.110 CACHE STRING "Verilator version for RTL simulation")
set(VCS vcs-2020.12 CACHE STRING "VCS version for RTL simulations" )

set(num_threads  1  CACHE STRING "Number of active cores")

set(banshee_stack_size 16777216 CACHE STRING "Stack size of banshee threads")

OPTION(banshee_simulation "Optimize binary for banshee simulation" OFF)
OPTION(gvsoc_simulation "adapt preprocessor macro for gvsoc simulation" OFF)
if(banshee_simulation)
  add_compile_definitions(BANSHEE_SIMULATION)
endif()
if(gvsoc_simulation)
	add_compile_definitions(GVSOC_SIMULATION)
endif()

#########################
##  Utility Functions  ##
#########################

macro(print_simulation_config)
	message(STATUS "============================= Simulation Configuration ============================")
	message(STATUS "[Simulator]   QuestaSim              = " ${QUESTA})
	message(STATUS "[Simulator]   Verilator              = " ${VERILATOR})
	message(STATUS "[Simulator]   VCS                    = " ${VCS})
	message(STATUS "[Simulator]   banshee_simulation     = " ${banshee_simulation})
	message(STATUS "[Simulator]   banshee_configuration  = " ${BANSHEE_CONFIG})
	message(STATUS "[Simulator]   gvsoc_simulation       = " ${gvsoc_simulation})
	message(STATUS "[Simulator]   banshee_stack_size     = " ${banshee_stack_size})
	message(STATUS "[Simulator]   num_threads            = " ${num_threads})
	message(STATUS "================================================================================")
	message(STATUS "")
endmacro()

macro(add_banshee_simulation name)
	if(NOT DEFINED ENV{BANSHEE_INSTALL_DIR})
		message(FATAL_ERROR "Environment variable BANSHEE_INSTALL_DIR not set")
	endif()
	set(BANSHEE_EXECUTABLE "$ENV{BANSHEE_INSTALL_DIR}/banshee")
    add_custom_target(banshee_${name}
	DEPENDS ${name}
	COMMAND RUST_MIN_STACK=${banshee_stack_size} ${BANSHEE_EXECUTABLE}
	--num-cores=${num_threads}
	--num-clusters=1
	--latency
	--configuration
	${BANSHEE_CONFIG}
	${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name} || true
	COMMENT "Simulating deeploytest with banshe"
	POST_BUILD
	USES_TERMINAL
	VERBATIM
    )
endmacro()

# Function to create the flags for hyperflash
# Saves the result into the `out_var`
function(gvsoc_flags_add_files_to_hyperflash out_var files_var)
	# LMACAN: The double variable expansion is needed because the first
	# expansion gets us the variable name, the second one actual list elements
	set(flags ${${files_var}})
	list(TRANSFORM flags PREPEND "--flash-property=")
	list(TRANSFORM flags APPEND "@hyperflash:readfs:files")
	set(${out_var} ${flags} PARENT_SCOPE)
endfunction()

# The macro creates a new gvsoc_<name> cmake target which executes the final
# binary on the gvsoc simulator. To give extra flags to the gvsoc command, set
# the GVSOC_EXTRA_FLAGS variable.
macro(add_gvsoc_emulation name target)
	if(NOT DEFINED ENV{GVSOC_INSTALL_DIR})
		message(FATAL_ERROR "Environment variable GVSOC_INSTALL_DIR not set")
	endif()
	set(GVSOC_WORKDIR ${CMAKE_BINARY_DIR}/gvsoc_workdir)
	make_directory(${GVSOC_WORKDIR})
	set(GVSOC_EXECUTABLE "$ENV{GVSOC_INSTALL_DIR}/bin/gvsoc")
	set(GVSOC_BINARY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${name}")
	add_custom_target(gvsoc_${name}
		DEPENDS ${name}
		COMMAND ${GVSOC_EXECUTABLE} --target=${target} --binary ${GVSOC_BINARY} --work-dir=${GVSOC_WORKDIR} ${GVSOC_EXTRA_FLAGS} image flash run
		COMMENT "Simulating deeploytest ${name} with gvsoc for the target ${target}"
		POST_BUILD
		USES_TERMINAL
		VERBATIM
	)
endmacro()
