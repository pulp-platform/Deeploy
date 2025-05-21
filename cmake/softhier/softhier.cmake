# SoftHier Configuration 
set(num_cluster_x        4   CACHE STRING "Number of clusters on x-axis")
set(num_cluster_y        4   CACHE STRING "Number of clusters on y-axis")

set(cluster_tcdm_base       0           CACHE STRING "") # 0x0000_0000
set(hbm_start_base          3221225472  CACHE STRING "") # 0xc000_0000

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

    CLUSTER_TCDM_BASE=${cluster_tcdm_base}
    HBM_START_BASE=${hbm_start_base}
)

add_compile_definitions(
    DEFAULT_MEM=${default_hbm}
    DEFAULT_L1=${default_l1}
    DEFAULT_HBM=${default_hbm}
)