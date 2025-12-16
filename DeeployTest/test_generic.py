# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import List

import pytest

from testUtils.pytestRunner import (
    DeeployTestConfig,
    get_test_paths,
    run_complete_test,
    get_worker_id,
)

KERNEL_TESTS = [
    "Adder",
    "MultIO",
    "test1DConvolution",
    "test2DConvolution",
    "test1DDWConvolution",
    "test2DDWConvolution",
    "test1DPad",
    "test2DPad",
    "testGEMM",
    "testMatMul",
    "testMatMulAdd",
    "testMaxPool",
    "testRQConv",
    "testRQMatMul",
    "testReduceSum",
    "testReduceMean",
    "testSlice",
    "testRequantizedDWConv",
    "test2DRequantizedConv",
    "iSoftmax",
    "testFloatAdder",
    "testFloatGEMM",
    "testFloat2DConvolution",
    "testFloat2DConvolutionBias",
    "testFloat2DConvolutionZeroBias",
    "testFloatLayerNorm",
    "testFloatDiv",
    "testFloat2DDWConvolution",
    "testFloat2DDWConvolutionBias",
    "testFloat2DDWConvolutionZeroBias",
    "testFloatRelu",
    "testFloatMaxPool",
    "testFloatMatmul",
    "testFloatReshapeWithSkipConnection",
    "testFloatSoftmax",
    "testFloatTranspose",
    "testFloatMul",
    "testFloatPowScalar",
    "testFloatPowVector",
    "testFloatSqrt",
    "testFloatRMSNorm",
    "Quant",
    "Dequant",
    "QuantizedLinear",
]

MODEL_TESTS = [
    "simpleRegression",
    "WaveFormer",
    "simpleCNN",
    "ICCT",
    "ICCT_ITA",
    "ICCT_8",
    "ICCT_ITA_8",
    "miniMobileNet",
    "miniMobileNetv2",
    "CCT/CCT_1_16_16_8",
    "CCT/CCT_2_32_32_128_Opset20",
    "testFloatDemoTinyViT",
    "Autoencoder1D",
]

def create_test_config(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args):
    """
    Create DeeployTestConfig for a specific test.
    
    Args:
        test_name: Name of the test
        deeploy_test_dir: Base DeeployTest directory (from fixture)
        toolchain: Toolchain to use - LLVM/GCC (from fixture)
        toolchain_dir: Path to toolchain installation (from fixture)
        cmake_args: Additional CMake arguments (from fixture)
        
    Returns:
        DeeployTestConfig instance
    """
    platform = "Generic"
    test_dir = f"Tests/{test_name}"
    
    gen_dir, test_dir_abs, test_name_clean = get_test_paths(
        test_dir, platform, base_dir=deeploy_test_dir
    )
    
    worker_id = get_worker_id()
    build_dir = str(Path(deeploy_test_dir) / f"TEST_{platform.upper()}" / f"build_{worker_id}")
    
    config = DeeployTestConfig(
        test_name=test_name_clean,
        test_dir=test_dir_abs,
        platform=platform,
        simulator="host",
        tiling=False,
        gen_dir=gen_dir,
        build_dir=build_dir,
        toolchain=toolchain,
        toolchain_install_dir=toolchain_dir,
        cmake_args=cmake_args,
    )
    
    return config

def run_and_assert_test(test_name, config, skipgen, skipsim):
    """
    Shared helper function to run a test and assert its results.
    
    Args:
        test_name: Name of the test
        config: DeeployTestConfig instance
        skipgen: Whether to skip network generation
        skipsim: Whether to skip simulation
    """
    # Run the complete test
    result = run_complete_test(config, skipgen=skipgen, skipsim=skipsim)
    
    # Assert results
    assert result.success, (
        f"Test {test_name} failed with {result.error_count} errors out of {result.total_count}\n"
        f"Output:\n{result.stdout}"
    )
    
    if result.error_count >= 0:  # Valid parse
        assert result.error_count == 0, (
            f"Found {result.error_count} errors out of {result.total_count} tests"
        )
    
@pytest.mark.generic
@pytest.mark.kernels
@pytest.mark.parametrize("test_name", KERNEL_TESTS, ids=KERNEL_TESTS)
def test_generic_kernels(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    config = create_test_config(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args)
    run_and_assert_test(test_name, config, skipgen, skipsim)

@pytest.mark.generic
@pytest.mark.models
@pytest.mark.parametrize("test_name", MODEL_TESTS, ids=MODEL_TESTS)
def test_model(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args, skipgen, skipsim) -> None:
    config = create_test_config(test_name, deeploy_test_dir, toolchain, toolchain_dir, cmake_args)
    run_and_assert_test(test_name, config, skipgen, skipsim)
