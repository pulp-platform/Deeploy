# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import pytest
from testUtils.pytestRunner import create_test_config, run_and_assert_test

# Import platform-specific test configurations
from test_generic_config import KERNEL_TESTS as GENERIC_KERNEL_TESTS
from test_generic_config import MODEL_TESTS as GENERIC_MODEL_TESTS
from test_cortexm_config import KERNEL_TESTS as CORTEXM_KERNEL_TESTS
from test_cortexm_config import MODEL_TESTS as CORTEXM_MODEL_TESTS
from test_siracusa_config import KERNEL_TESTS as SIRACUSA_KERNEL_TESTS
from test_siracusa_config import MODEL_TESTS as SIRACUSA_MODEL_TESTS
from test_siracusa_config import DEFAULT_CORES as SIRACUSA_DEFAULT_CORES
from test_siracusa_tiled_config import (
    L2_SINGLEBUFFER_KERNELS,
    L2_DOUBLEBUFFER_KERNELS,
    L2_SINGLEBUFFER_MODELS,
    L2_DOUBLEBUFFER_MODELS,
    L3_SINGLEBUFFER_MODELS,
    L3_DOUBLEBUFFER_MODELS,
)


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
}


### Markers summary ###
# Platform markers:
#   generic: tests from the generic platform
#   cortexm: tests from the cortex-m (QEMU-ARM) platform
#   siracusa: tests from the Siracusa platform (untiled)
#   siracusa_tiled: tests from the Siracusa platform (tiled)
# Test type markers:
#   kernels: single kernel (or single layer) tests
#   models: full model (multiple layer) tests
# Configuration markers (tiled platforms):
#   singlebuffer: single-buffer tests
#   doublebuffer: double-buffer tests
#   l2: L2 default memory level
#   l3: L3 default memory level


@pytest.mark.generic
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", GENERIC_KERNEL_TESTS, ids = GENERIC_KERNEL_TESTS)
def test_generic_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    """Test Generic platform kernel tests."""
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
    """Test Generic platform model tests."""
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
    """Test Cortex-M platform kernel tests."""
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
    """Test Cortex-M platform model tests."""
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


### Siracusa Platform Tests ###


@pytest.mark.siracusa
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", SIRACUSA_KERNEL_TESTS, ids = SIRACUSA_KERNEL_TESTS)
def test_siracusa_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    """Test Siracusa platform kernel tests (untiled)."""
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
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


@pytest.mark.siracusa
@pytest.mark.models
@pytest.mark.parametrize("test_name", SIRACUSA_MODEL_TESTS, ids = SIRACUSA_MODEL_TESTS)
def test_siracusa_models(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    """Test Siracusa platform model tests (untiled)."""
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
    )
    run_and_assert_test(test_name, config, skipgen, skipsim)


### Siracusa Tiled Platform Tests ###


@pytest.mark.siracusa_tiled
@pytest.mark.kernels
@pytest.mark.singlebuffer
@pytest.mark.l2
@pytest.mark.parametrize(
    "test_params",
    generate_test_params(L2_SINGLEBUFFER_KERNELS, "L2-singlebuffer"),
    ids = param_id,
)
def test_siracusa_tiled_kernels_l2_singlebuffer(
    test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim
) -> None:
    """Test Siracusa tiled kernel tests (L2, single-buffer)."""
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
def test_siracusa_tiled_kernels_l2_doublebuffer(
    test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim
) -> None:
    """Test Siracusa tiled kernel tests (L2, double-buffer)."""
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
def test_siracusa_tiled_models_l2_singlebuffer(
    test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim
) -> None:
    """Test Siracusa tiled model tests (L2, single-buffer)."""
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
def test_siracusa_tiled_models_l2_doublebuffer(
    test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim
) -> None:
    """Test Siracusa tiled model tests (L2, double-buffer)."""
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
def test_siracusa_tiled_models_l3_singlebuffer(
    test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim
) -> None:
    """Test Siracusa tiled model tests (L3, single-buffer)."""
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
def test_siracusa_tiled_models_l3_doublebuffer(
    test_params, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim
) -> None:
    """Test Siracusa tiled model tests (L3, double-buffer)."""
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
