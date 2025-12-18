# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
Central test file for all platforms.

This file defines the test functions with markers for all supported platforms.
Each platform's test lists are imported from their respective config files.
"""

import pytest
from testUtils.pytestRunner import create_test_config, run_and_assert_test

# Import platform-specific test configurations
from test_generic_config import KERNEL_TESTS as GENERIC_KERNEL_TESTS
from test_generic_config import MODEL_TESTS as GENERIC_MODEL_TESTS
from test_cortexm_config import KERNEL_TESTS as CORTEXM_KERNEL_TESTS
from test_cortexm_config import MODEL_TESTS as CORTEXM_MODEL_TESTS


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
# generic: tests from the generic platform
# cortexm: tests from the cortex-m (QEMU-ARM) platform
# kernels: single kernel (or single layer) tests
# models: full model (multiple layer) tests


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
