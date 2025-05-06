# SoftHier Configuration 

# cluster-level config
set(num_cluster_x        4   CACHE STRING "Number of clusters on x-axis")
set(num_cluster_y        4   CACHE STRING "Number of clusters on y-axis")
set(num_core_per_cluster 3   CACHE STRING "Number of Snitch cores per cluster")

set(cluster_tcdm_bank_width  32    CACHE STRING "L1 TCDM bank width (in bit)")
set(cluster_tcdm_bank_nb     128   CACHE STRING "L1 TCDM bank per cluster")   # check with Chi

set(cluster_tcdm_base       0           CACHE STRING "") # 0x0000_0000
set(cluster_tcdm_size       1048576     CACHE STRING "") # 0x0010_0000
set(cluster_tcdm_remote     805306368   CACHE STRING "") # 0x3000_0000

set(cluster_stack_base      268435456   CACHE STRING "") # 0x1000_0000
set(cluster_stack_size      131072      CACHE STRING "") # 0x0002_0000

set(cluster_zomem_base      402653184   CACHE STRING "") # 0x1800_0000
set(cluster_zomem_size      131072      CACHE STRING "") # 0x0002_0000

set(cluster_reg_base        536870912   CACHE STRING "") # 0x2000_0000
set(cluster_reg_size        512         CACHE STRING "") # 0x0000_0200

# Spatz vector unit
set(spatz_attaced_core_list 1           CACHE STRING "") 
set(spatz_num_vlsu_port     4           CACHE STRING "") 
set(spatz_num_function_unit 4           CACHE STRING "") 

# RedMule
set(redmule_ce_height       128         CACHE STRING "")
set(redmule_ce_width        32          CACHE STRING "")
set(redmule_ce_pipe         3           CACHE STRING "")
set(redmule_elem_size       2           CACHE STRING "")
set(redmule_queue_depth     1           CACHE STRING "")
set(redmule_reg_base        537001984   CACHE STRING "") # 0x2002_0000
set(redmule_reg_size        512         CACHE STRING "") # 0x0000_0200

# IDMA
set(idma_outstand_txn       16          CACHE STRING "")
set(idma_outstand_burst     256         CACHE STRING "")

# HBM
set(hbm_start_base          3221225472  CACHE STRING "") # 0xc000_0000
set(hbm_node_addr_space     2097152     CACHE STRING "") # 0x0020_0000
set(num_node_per_ctrl       1           CACHE STRING "")
set(hbm_chan_placement      4           CACHE STRING "") # bowwang: do we need this?

# NoC
set(noc_outstanding         64          CACHE STRING "")
set(noc_link_width          512         CACHE STRING "")

# System
set(instruction_mem_base    2147483648  CACHE STRING "") # 0x8000_0000
set(instruction_mem_size    65536       CACHE STRING "") # 0x0001_0000

set(soc_register_base       2415919104  CACHE STRING "") # 0x9000_0000
set(soc_register_size       65536       CACHE STRING "") # 0x0001_0000
set(soc_register_eoc        2415919104  CACHE STRING "") # 0x9000_0000
set(soc_register_wakeup     2415919108  CACHE STRING "") # 0x9000_0004

# Sync
set(sync_base               1073741824  CACHE STRING "") # 0x4000_0000
set(sync_interleave         128         CACHE STRING "") # 0x0000_0080
set(sync_special_mem        64          CACHE STRING "") # 0x0000_0040

# Default memory level
set(default_l1              0           CACHE STRING "")
set(default_hbm             1           CACHE STRING "")

add_compile_definitions(
    DEEPLOY_SOFTHIER_PLATFORM
)

# Cluster
add_compile_definitions(
    NUM_CLUSTER_X=${num_cluster_x}
    NUM_CLUSTER_Y=${num_cluster_y}
    NUM_CORE_PER_CLUSTER=${num_core_per_cluster}

    CLUSTER_TCDM_BANK_WIDTH=${cluster_tcdm_bank_width}
    CLUSTER_TCDM_BANK_NB=${cluster_tcdm_bank_nb}

    CLUSTER_TCDM_BASE=${cluster_tcdm_base}
    CLUSTER_TCDM_SIZE=${cluster_tcdm_size}
    CLUSTER_TCDM_REMOTE=${cluster_tcdm_remote}

    CLUSTER_STACK_BASE=${cluster_stack_base}
    CLUSTER_STACK_SIZE=${cluster_stack_size}

    CLUSTER_ZOMEM_BASE=${cluster_zomem_base}
    CLUSTER_ZOMEM_SIZE=${cluster_zomem_size}

    CLUSTER_REG_BASE=${cluster_reg_base}
    CLUSTER_REG_SIZE=${cluster_reg_size}
)

# acc
add_compile_definitions(
    SPATZ_ATTACED_CORE_LIST=${spatz_attaced_core_list}
    SPATZ_NUM_VLSU_PORT=${spatz_num_vlsu_port}
    SPATZ_NUM_FUNCTION_UNIT=${spatz_num_function_unit}

    REDMULE_CE_HEIGHT=${redmule_ce_height}
    REDMULE_CE_WIDTH=${redmule_ce_width}
    REDMULE_CE_PIPE=${redmule_ce_pipe}
    REDMULE_ELEM_SIZE=${redmule_elem_size}
    REDMULE_QUEUE_DEPTH=${redmule_queue_depth}
    REDMULE_REG_BASE=${redmule_reg_base}
    REDMULE_REG_SIZE=${redmule_reg_size}
)

# periph
add_compile_definitions(
    IDMA_OUTSTAND_TXN=${idma_outstand_txn}
    IDMA_OUTSTAND_BURST=${idma_outstand_burst}

    HBM_START_BASE=${hbm_start_base}
    HBM_NODE_ADDR_SPACE=${hbm_node_addr_space}
    NUM_NODE_PER_CTRL=${num_node_per_ctrl}
    HBM_CHAN_PLACEMENT=${hbm_chan_placement}

    NOC_OUTSTANDING=${noc_outstanding}
    NOC_LINK_WIDTH=${noc_link_width}
)

# system
add_compile_definitions(
    INSTRUCTION_MEM_BASE=${instruction_mem_base}
    INSTRUCTION_MEM_SIZE=${instruction_mem_size}

    SOC_REGISTER_BASE=${soc_register_base}
    SOC_REGISTER_SIZE=${soc_register_size}
    SOC_REGISTER_EOC=${soc_register_eoc}
    SOC_REGISTER_WAKEUP=${soc_register_wakeup}

    SYNC_BASE=${sync_base}
    SYNC_INTERLEAVE=${sync_interleave}
    SYNC_SPECIAL_MEM=${sync_special_mem}
)

add_compile_definitions(
    DEFAULT_MEM=${default_hbm}
    DEFAULT_L1=${default_l1}
    DEFAULT_HBM=${default_hbm}
)