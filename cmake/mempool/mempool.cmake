#############################
##  Address configuration  ##
#############################

# Boot address (in dec)
set(boot_addr  2684354560   CACHE STRING "Boot address (in dec)") # A0000000

# L2 memory configuration (in hex)
set(l2_base   2147483648    CACHE STRING "L2 Memory Base (in dec)") # 80000000
set(l2_size   4194304       CACHE STRING "L2 Memory Size (in dec)")    # 400000
set(l2_banks  4             CACHE STRING "Number of L2 banks")

# Size of sequential memory per core (in bytes)
# (must be a power of two)
set(seq_mem_size  1024      CACHE STRING "Size of sequential memory per core (in bytes, must be a power of two)")

# Size of stack in sequential memory per core (in bytes)
set(stack_size  1024        CACHE STRING "Size of stack in sequential memory per core (in bytes)")

#########################
##  AXI configuration  ##
#########################
# AXI bus data width (in bits)
set(axi_data_width  512     CACHE STRING "AXI bus data width (in bits)")

# Read-only cache line width in AXI interconnect (in bits)
set(ro_line_width  512      CACHE STRING "Read-only cache line width in AXI interconnect (in bits)")

# Number of DMA backends in each group
set(dmas_per_group  4       CACHE STRING "Number of DMA backends in each group")

#############################
##  Xqueues configuration  ##
#############################

# XQueue extension's queue size in each memory bank (in words)
set(xqueue_size  0          CACHE STRING "XQueue extension's queue size in each memory bank (in words)")

################################
##  Optional functionalities  ##
################################

# Enable the XpulpIMG extension
set(xpulpimg  0             CACHE STRING "Enable the XpulpIMG extension")

##################
##  Simulation  ##
##################

set(BANSHEE_CONFIG ${CMAKE_CURRENT_LIST_DIR}/mempool.yaml CACHE INTERNAL "source_list")

###############
##  MemPool  ##
###############

# Number of cores
set(num_cores  256          CACHE STRING "Number of cores")

set(num_eff_cores 256       CACHE STRING "Number of effective cores")

# Number of groups
set(num_groups  4           CACHE STRING "Number of groups")

# Number of cores per MemPool tile
set(num_cores_per_tile  4   CACHE STRING "Number of cores per MemPool tile")

# L1 scratchpad banking factor
set(banking_factor  4       CACHE STRING "L1 scratchpad banking factor")

# Radix for hierarchical AXI interconnect
set(axi_hier_radix  20      CACHE STRING "Radix for hierarchical AXI interconnect")

# Number of AXI masters per group
set(axi_masters_per_group 1 CACHE STRING "Number of AXI masters per group")

math_shell("${num_cores} / ${num_groups}" num_cores_per_group)
math_shell("(${num_cores} / ${num_groups}) / ${num_cores_per_tile}" num_tiles_per_group)
math_shell("log(${num_cores_per_tile}) / log(2)" log2_num_cores_per_tile)
math_shell("log(${seq_mem_size}) / log(2)" log2_seq_mem_size)
math_shell("log(${stack_size}) / log(2)" log2_stack_size)

add_compile_definitions(
    DEEPLOY_MEMPOOL_PLATFORM
)

add_compile_definitions(

    PRINTF_DISABLE_SUPPORT_FLOAT
    PRINTF_DISABLE_SUPPORT_LONG_LONG
    PRINTF_DISABLE_SUPPORT_PTRDIFF_T

    NUM_CORES=${num_cores}
    NUM_EFF_CORES=${num_eff_cores}

    NUM_THREADS=${num_threads}
    NUM_GROUPS=${num_groups}
    NUM_CORES_PER_TILE=${num_cores_per_tile}
    LOG2_NUM_CORES_PER_TILE=${log2_num_cores_per_tile}
    BANKING_FACTOR=${banking_factor}
    NUM_CORES_PER_GROUP=${num_cores_per_group}
    NUM_TILES_PER_GROUP=${num_tiles_per_group}

    BOOT_ADDR=${boot_addr}
    L2_BASE=${l2_base}
    L2_SIZE=${l2_size}
    LOG2_SEQ_MEM_SIZE=${log2_seq_mem_size}
    SEQ_MEM_SIZE=${seq_mem_size}
    STACK_SIZE=${stack_size}
    LOG2_STACK_SIZE=${log2_stack_size}
    XQUEUE_SIZE=${xqueue_size}
)
