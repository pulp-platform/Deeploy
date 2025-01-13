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
    add_custom_target(banshee_${name}
	DEPENDS ${name}
	COMMAND RUST_MIN_STACK=${banshee_stack_size} banshee
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
