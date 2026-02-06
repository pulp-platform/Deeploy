# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import pytest
# Import platform-specific test configurations
from test_chimera_config import KERNEL_TESTS as CHIMERA_KERNEL_TESTS
from test_chimera_config import MODEL_TESTS as CHIMERA_MODEL_TESTS
from test_cortexm_config import KERNEL_TESTS as CORTEXM_KERNEL_TESTS
from test_cortexm_config import MODEL_TESTS as CORTEXM_MODEL_TESTS
from test_gap9_config import DEFAULT_NUM_CORES as GAP9_DEFAULT_NUM_CORES
from test_gap9_config import KERNEL_TESTS as GAP9_KERNEL_TESTS
from test_gap9_config import MODEL_TESTS as GAP9_MODEL_TESTS
from test_gap9_tiled_config import DEFAULT_CORES as GAP9_TILED_DEFAULT_CORES
from test_gap9_tiled_config import L2_DOUBLEBUFFER_KERNELS as GAP9_L2_DOUBLEBUFFER_KERNELS
from test_gap9_tiled_config import L2_DOUBLEBUFFER_MODELS as GAP9_L2_DOUBLEBUFFER_MODELS
from test_gap9_tiled_config import L2_SINGLEBUFFER_KERNELS as GAP9_L2_SINGLEBUFFER_KERNELS
from test_gap9_tiled_config import L2_SINGLEBUFFER_MODELS as GAP9_L2_SINGLEBUFFER_MODELS
from test_gap9_tiled_config import L3_DOUBLEBUFFER_MODELS as GAP9_L3_DOUBLEBUFFER_MODELS
from test_gap9_tiled_config import L3_SINGLEBUFFER_MODELS as GAP9_L3_SINGLEBUFFER_MODELS
from test_generic_config import KERNEL_TESTS as GENERIC_KERNEL_TESTS
from test_generic_config import MODEL_TESTS as GENERIC_MODEL_TESTS
from test_mempool_config import DEFAULT_NUM_THREADS as MEMPOOL_DEFAULT_NUM_THREADS
from test_mempool_config import KERNEL_TESTS as MEMPOOL_KERNEL_TESTS
from test_mempool_config import MODEL_TESTS as MEMPOOL_MODEL_TESTS
from test_siracusa_config import DEFAULT_CORES as SIRACUSA_DEFAULT_CORES
from test_siracusa_config import KERNEL_TESTS as SIRACUSA_KERNEL_TESTS
from test_siracusa_config import MODEL_TESTS as SIRACUSA_MODEL_TESTS
from test_siracusa_neureka_tiled_config import DEFAULT_CORES as NEUREKA_DEFAULT_CORES
from test_siracusa_neureka_tiled_config import L2_DOUBLEBUFFER_KERNELS as NEUREKA_L2_DOUBLEBUFFER_KERNELS
from test_siracusa_neureka_tiled_config import L2_SINGLEBUFFER_KERNELS as NEUREKA_L2_SINGLEBUFFER_KERNELS
from test_siracusa_neureka_tiled_config import L2_SINGLEBUFFER_KERNELS_WMEM as NEUREKA_L2_SINGLEBUFFER_KERNELS_WMEM
from test_siracusa_neureka_tiled_config import L3_DOUBLEBUFFER_MODELS as NEUREKA_L3_DOUBLEBUFFER_MODELS
from test_siracusa_neureka_tiled_config import L3_DOUBLEBUFFER_MODELS_WMEM as NEUREKA_L3_DOUBLEBUFFER_MODELS_WMEM
from test_siracusa_neureka_tiled_config import L3_SINGLEBUFFER_MODELS as NEUREKA_L3_SINGLEBUFFER_MODELS
from test_siracusa_tiled_config import L2_DOUBLEBUFFER_KERNELS, L2_DOUBLEBUFFER_MODELS, L2_SINGLEBUFFER_KERNELS, \
    L2_SINGLEBUFFER_MODELS, L3_DOUBLEBUFFER_MODELS, L3_SINGLEBUFFER_MODELS
from test_snitch_config import DEFAULT_NUM_CORES as SNITCH_DEFAULT_NUM_CORES
from test_snitch_config import KERNEL_TESTS as SNITCH_KERNEL_TESTS
from test_snitch_config import MODEL_TESTS as SNITCH_MODEL_TESTS
from test_snitch_tiled_config import L2_SINGLEBUFFER_KERNELS as SNITCH_L2_SINGLEBUFFER_KERNELS
from test_snitch_tiled_config import L2_SINGLEBUFFER_MODELS as SNITCH_L2_SINGLEBUFFER_MODELS
from test_softhier_config import DEFAULT_NUM_CLUSTERS as SOFTHIER_DEFAULT_NUM_CLUSTERS
from test_softhier_config import KERNEL_TESTS as SOFTHIER_KERNEL_TESTS
from test_softhier_config import MODEL_TESTS as SOFTHIER_MODEL_TESTS
from testUtils.pytestRunner import create_test_config, run_and_assert_test


def generate_test_params(test_dict, config_name):
    """
    Generate test parameters from a dictionary of test names to L1 values.

    Args:
        test_dict: Dictionary mapping test_name -> list of L1 values
        config_name: Configuration name for test ID (e.g., "L2-singlebuffer")

    Returns:
        List of (test_name, l1_value, config_name) tuples
    """
    params = []
    for test_name, l1_values in test_dict.items():
        for l1 in l1_values:
            params.append((test_name, l1, config_name))
    return params


def param_id(param):
    """Generate test ID from parameter tuple."""
    test_name, l1, config = param
    return f"{test_name}-{l1}-{config}"


### Platform Configuration ###
PLATFORM_CONFIGS = {
    "generic": {
        "platform": "Generic",
        "simulator": "host",
        "kernel_tests": GENERIC_KERNEL_TESTS,
        "model_tests": GENERIC_MODEL_TESTS,
    },
    "cortexm": {
        "platform": "QEMU-ARM",
        "simulator": "qemu",
        "kernel_tests": CORTEXM_KERNEL_TESTS,
        "model_tests": CORTEXM_MODEL_TESTS,
    },
    "mempool": {
        "platform": "MemPool",
        "simulator": "banshee",
        "kernel_tests": MEMPOOL_KERNEL_TESTS,
        "model_tests": MEMPOOL_MODEL_TESTS,
        "default_num_threads": MEMPOOL_DEFAULT_NUM_THREADS,
    },
    "chimera": {
        "platform": "Chimera",
        "simulator": "gvsoc",
        "kernel_tests": CHIMERA_KERNEL_TESTS,
        "model_tests": CHIMERA_MODEL_TESTS,
    },
    "softhier": {
        "platform": "SoftHier",
        "simulator": "gvsoc",
        "kernel_tests": SOFTHIER_KERNEL_TESTS,
        "model_tests": SOFTHIER_MODEL_TESTS,
        "default_num_clusters": SOFTHIER_DEFAULT_NUM_CLUSTERS,
    },
    "snitch": {
        "platform": "Snitch",
        "simulator": "gvsoc",
        "kernel_tests": SNITCH_KERNEL_TESTS,
        "model_tests": SNITCH_MODEL_TESTS,
        "default_num_cores": SNITCH_DEFAULT_NUM_CORES,
    },
    "gap9": {
        "platform": "GAP9",
        "simulator": "gvsoc",
        "kernel_tests": GAP9_KERNEL_TESTS,
        "model_tests": GAP9_MODEL_TESTS,
        "default_num_cores": GAP9_DEFAULT_NUM_CORES,
    },
}

### Markers summary ###
# Platform markers:
#   generic: tests from the generic platform
#   cortexm: tests from the cortex-m (QEMU-ARM) platform
#   mempool: tests from the MemPool platform
#   chimera: tests from the Chimera platform
#   softhier: tests from the SoftHier platform
#   snitch: tests from the Snitch platform (untiled)
#   snitch_tiled: tests from the Snitch platform (tiled)
#   siracusa: tests from the Siracusa platform (untiled)
#   siracusa_tiled: tests from the Siracusa platform (tiled)
#   siracusa_neureka_tiled: tests from the Siracusa + Neureka platform (tiled)
#   gap9: tests from the GAP9 platform (untiled)
#   gap9_tiled: tests from the GAP9 platform (tiled)
# Test type markers:
#   kernels: single kernel (or single layer) tests
#   models: full model (multiple layer) tests
# Configuration markers (tiled platforms):
#   singlebuffer: single-buffer tests
#   doublebuffer: double-buffer tests
#   l2: L2 default memory level
#   l3: L3 default memory level
#   wmem: with Neureka weight memory enabled


@pytest.mark.generic
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", GENERIC_KERNEL_TESTS, ids = GENERIC_KERNEL_TESTS)
def test_generic_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["generic"]
    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.generic
@pytest.mark.models
@pytest.mark.parametrize("test_name", GENERIC_MODEL_TESTS, ids = GENERIC_MODEL_TESTS)
def test_generic_models(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["generic"]
    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.cortexm
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", CORTEXM_KERNEL_TESTS, ids = CORTEXM_KERNEL_TESTS)
def test_cortexm_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["cortexm"]
    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.cortexm
@pytest.mark.models
@pytest.mark.parametrize("test_name", CORTEXM_MODEL_TESTS, ids = CORTEXM_MODEL_TESTS)
def test_cortexm_models(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["cortexm"]
    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.mempool
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", MEMPOOL_KERNEL_TESTS, ids = MEMPOOL_KERNEL_TESTS)
def test_mempool_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["mempool"]

    # Add MemPool-specific CMake args for number of threads
    mempool_cmake_args = cmake_args + [f"num_threads={platform_config['default_num_threads']}"]

    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = mempool_cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.mempool
@pytest.mark.models
@pytest.mark.parametrize("test_name", MEMPOOL_MODEL_TESTS, ids = MEMPOOL_MODEL_TESTS)
def test_mempool_models(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["mempool"]

    # Add MemPool-specific CMake args for number of threads
    mempool_cmake_args = cmake_args + [f"num_threads={platform_config['default_num_threads']}"]

    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = mempool_cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", SIRACUSA_KERNEL_TESTS, ids = SIRACUSA_KERNEL_TESTS)
def test_siracusa_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim,
                          profile_untiled) -> None:
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = False,
        cores = SIRACUSA_DEFAULT_CORES,
        profile_untiled = profile_untiled,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa
@pytest.mark.models
@pytest.mark.parametrize("test_name", SIRACUSA_MODEL_TESTS, ids = SIRACUSA_MODEL_TESTS)
def test_siracusa_models(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim,
                         profile_untiled) -> None:
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = False,
        cores = SIRACUSA_DEFAULT_CORES,
        profile_untiled = profile_untiled,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_tiled
@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(L2_SINGLEBUFFER_KERNELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_siracusa_tiled_kernels_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                                skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = SIRACUSA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_tiled
@pytest.mark.kernels
@pytest.mark.doublebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(L2_DOUBLEBUFFER_KERNELS, "L2-doublebuffer"),
    ids = param_id,
)
def test_siracusa_tiled_kernels_l2_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                                skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = SIRACUSA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_tiled
@pytest.mark.models
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(L2_SINGLEBUFFER_MODELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_siracusa_tiled_models_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                               skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = SIRACUSA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_tiled
@pytest.mark.models
@pytest.mark.doublebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(L2_DOUBLEBUFFER_MODELS, "L2-doublebuffer"),
    ids = param_id,
)
def test_siracusa_tiled_models_l2_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                               skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = SIRACUSA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_tiled
@pytest.mark.models
@pytest.mark.singlebuffer
@pytest.mark.l3
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(L3_SINGLEBUFFER_MODELS, "L3-singlebuffer"),
    ids = param_id,
)
def test_siracusa_tiled_models_l3_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                               skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = SIRACUSA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L3",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_tiled
@pytest.mark.models
@pytest.mark.doublebuffer
@pytest.mark.l3
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(L3_DOUBLEBUFFER_MODELS, "L3-doublebuffer"),
    ids = param_id,
)
def test_siracusa_tiled_models_l3_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                               skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = SIRACUSA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L3",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.chimera
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", CHIMERA_KERNEL_TESTS, ids = CHIMERA_KERNEL_TESTS)
def test_chimera_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["chimera"]
    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.softhier
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", SOFTHIER_KERNEL_TESTS, ids = SOFTHIER_KERNEL_TESTS)
def test_softhier_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["softhier"]

    # Add SoftHier-specific CMake args for number of clusters
    softhier_cmake_args = cmake_args + [f"num_clusters={platform_config['default_num_clusters']}"]

    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = softhier_cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.snitch
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", SNITCH_KERNEL_TESTS, ids = SNITCH_KERNEL_TESTS)
def test_snitch_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["snitch"]

    # Add Snitch-specific CMake args for number of cores
    snitch_cmake_args = cmake_args + [f"NUM_CORES={platform_config['default_num_cores']}"]

    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = snitch_cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.snitch
@pytest.mark.models
@pytest.mark.parametrize("test_name", SNITCH_MODEL_TESTS, ids = SNITCH_MODEL_TESTS)
def test_snitch_models(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["snitch"]
    snitch_cmake_args = cmake_args + [f"NUM_CORES={platform_config['default_num_cores']}"]
    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = snitch_cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.snitch_tiled
@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(SNITCH_L2_SINGLEBUFFER_KERNELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_snitch_tiled_kernels_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                              skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params

    # Add Snitch-specific CMake args
    snitch_cmake_args = cmake_args + [f"NUM_CORES={SNITCH_DEFAULT_NUM_CORES}"]

    config = create_test_config(
        test_name = test_name,
        platform = "Snitch",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = snitch_cmake_args,
        tiling = True,
        cores = SNITCH_DEFAULT_NUM_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.snitch_tiled
@pytest.mark.models
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(SNITCH_L2_SINGLEBUFFER_MODELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_snitch_tiled_models_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                             skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    snitch_cmake_args = cmake_args + [f"NUM_CORES={SNITCH_DEFAULT_NUM_CORES}"]
    config = create_test_config(
        test_name = test_name,
        platform = "Snitch",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = snitch_cmake_args,
        tiling = True,
        cores = SNITCH_DEFAULT_NUM_CORES,
        l1 = l1,
        l2 = 4000000,
        default_mem_level = "L2",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_neureka_tiled
@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(NEUREKA_L2_SINGLEBUFFER_KERNELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_siracusa_neureka_tiled_kernels_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir,
                                                        cmake_args, skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa_w_neureka",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = NEUREKA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_neureka_tiled
@pytest.mark.kernels
@pytest.mark.doublebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(NEUREKA_L2_DOUBLEBUFFER_KERNELS, "L2-doublebuffer"),
    ids = param_id,
)
def test_siracusa_neureka_tiled_kernels_l2_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir,
                                                        cmake_args, skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa_w_neureka",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = NEUREKA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_neureka_tiled
@pytest.mark.models
@pytest.mark.singlebuffer
@pytest.mark.l3
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(NEUREKA_L3_SINGLEBUFFER_MODELS, "L3-singlebuffer"),
    ids = param_id,
)
def test_siracusa_neureka_tiled_models_l3_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir,
                                                       cmake_args, skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa_w_neureka",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = NEUREKA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L3",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_neureka_tiled
@pytest.mark.models
@pytest.mark.doublebuffer
@pytest.mark.l3
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(NEUREKA_L3_DOUBLEBUFFER_MODELS, "L3-doublebuffer"),
    ids = param_id,
)
def test_siracusa_neureka_tiled_models_l3_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir,
                                                       cmake_args, skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa_w_neureka",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = NEUREKA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L3",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_neureka_tiled
@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.wmem
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(NEUREKA_L2_SINGLEBUFFER_KERNELS_WMEM, "L2-singlebuffer-wmem"),
    ids = param_id,
)
def test_siracusa_neureka_tiled_kernels_l2_singlebuffer_wmem(test_params, deeploy_test_dir, toolchain, toolchain_dir,
                                                             cmake_args, skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa_w_neureka",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = NEUREKA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
        gen_args = ["--neureka-wmem"],
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa_neureka_tiled
@pytest.mark.models
@pytest.mark.doublebuffer
@pytest.mark.l3
@pytest.mark.wmem
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(NEUREKA_L3_DOUBLEBUFFER_MODELS_WMEM, "L3-doublebuffer-wmem"),
    ids = param_id,
)
def test_siracusa_neureka_tiled_models_l3_doublebuffer_wmem(test_params, deeploy_test_dir, toolchain, toolchain_dir,
                                                            cmake_args, skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params
    config = create_test_config(
        test_name = test_name,
        platform = "Siracusa_w_neureka",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = cmake_args,
        tiling = True,
        cores = NEUREKA_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L3",
        double_buffer = True,
        gen_args = ["--neureka-wmem"],
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", GAP9_KERNEL_TESTS, ids = GAP9_KERNEL_TESTS)
def test_gap9_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["gap9"]

    # Add GAP9-specific CMake args for number of cores
    gap9_cmake_args = cmake_args + [f"NUM_CORES={platform_config['default_num_cores']}"]

    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9
@pytest.mark.models
@pytest.mark.parametrize("test_name", GAP9_MODEL_TESTS, ids = GAP9_MODEL_TESTS)
def test_gap9_models(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    platform_config = PLATFORM_CONFIGS["gap9"]

    # Add GAP9-specific CMake args for number of cores
    gap9_cmake_args = cmake_args + [f"NUM_CORES={platform_config['default_num_cores']}"]

    config = create_test_config(
        test_name = test_name,
        platform = platform_config["platform"],
        simulator = platform_config["simulator"],
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9_tiled
@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(GAP9_L2_SINGLEBUFFER_KERNELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_gap9_tiled_kernels_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                            skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params

    # Add GAP9-specific CMake args
    gap9_cmake_args = cmake_args + [f"NUM_CORES={GAP9_TILED_DEFAULT_CORES}"]

    config = create_test_config(
        test_name = test_name,
        platform = "GAP9",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = True,
        cores = GAP9_TILED_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9_tiled
@pytest.mark.kernels
@pytest.mark.doublebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(GAP9_L2_DOUBLEBUFFER_KERNELS, "L2-doublebuffer"),
    ids = param_id,
)
def test_gap9_tiled_kernels_l2_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args,
                                            skipgen, skipsim) -> None:
    test_name, l1, config_name = test_params

    # Add GAP9-specific CMake args
    gap9_cmake_args = cmake_args + [f"NUM_CORES={GAP9_TILED_DEFAULT_CORES}"]

    config = create_test_config(
        test_name = test_name,
        platform = "GAP9",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = True,
        cores = GAP9_TILED_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9_tiled
@pytest.mark.models
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(GAP9_L2_SINGLEBUFFER_MODELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_gap9_tiled_models_l2_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                                           skipsim) -> None:
    test_name, l1, config_name = test_params

    # Add GAP9-specific CMake args
    gap9_cmake_args = cmake_args + [f"NUM_CORES={GAP9_TILED_DEFAULT_CORES}"]

    config = create_test_config(
        test_name = test_name,
        platform = "GAP9",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = True,
        cores = GAP9_TILED_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9_tiled
@pytest.mark.models
@pytest.mark.doublebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(GAP9_L2_DOUBLEBUFFER_MODELS, "L2-doublebuffer"),
    ids = param_id,
)
def test_gap9_tiled_models_l2_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                                           skipsim) -> None:
    test_name, l1, config_name = test_params

    # Add GAP9-specific CMake args
    gap9_cmake_args = cmake_args + [f"NUM_CORES={GAP9_TILED_DEFAULT_CORES}"]

    config = create_test_config(
        test_name = test_name,
        platform = "GAP9",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = True,
        cores = GAP9_TILED_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L2",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9_tiled
@pytest.mark.models
@pytest.mark.singlebuffer
@pytest.mark.l3
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(GAP9_L3_SINGLEBUFFER_MODELS, "L3-singlebuffer"),
    ids = param_id,
)
def test_gap9_tiled_models_l3_singlebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                                           skipsim) -> None:
    test_name, l1, config_name = test_params

    # Add GAP9-specific CMake args
    gap9_cmake_args = cmake_args + [f"NUM_CORES={GAP9_TILED_DEFAULT_CORES}"]

    config = create_test_config(
        test_name = test_name,
        platform = "GAP9",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = True,
        cores = GAP9_TILED_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L3",
        double_buffer = False,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.gap9_tiled
@pytest.mark.models
@pytest.mark.doublebuffer
@pytest.mark.l3
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(GAP9_L3_DOUBLEBUFFER_MODELS, "L3-doublebuffer"),
    ids = param_id,
)
def test_gap9_tiled_models_l3_doublebuffer(test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen,
                                           skipsim) -> None:
    test_name, l1, config_name = test_params

    # Add GAP9-specific CMake args
    gap9_cmake_args = cmake_args + [f"NUM_CORES={GAP9_TILED_DEFAULT_CORES}"]

    config = create_test_config(
        test_name = test_name,
        platform = "GAP9",
        simulator = "gvsoc",
        deeploy_test_dir = deeploy_test_dir,
        toolchain = toolchain,
        toolchain_dir = toolchain_dir,
        cmake_args = gap9_cmake_args,
        tiling = True,
        cores = GAP9_TILED_DEFAULT_CORES,
        l1 = l1,
        default_mem_level = "L3",
        double_buffer = True,
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)
