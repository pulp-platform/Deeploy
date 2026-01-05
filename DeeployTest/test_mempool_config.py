# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""
Test configuration for MemPool platform.

This module defines the test lists and default parameters for MemPool platform tests.
"""

# Default number of threads for MemPool
DEFAULT_NUM_THREADS = 16

# Kernel tests (individual operators)
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
    "testRQGEMM",
    "testRQMatMul",
    "testReduceSum",
    "testReduceMean",
    "testSlice",
    "testRequantizedDWConv",
    "test2DRequantizedConv",
]

# Model tests (full networks)
MODEL_TESTS = [
    "simpleRegression",
    "simpleCNN",
    "ICCT",
    "ICCT_ITA",
    "ICCT_8",
    "miniMobileNet",
    "miniMobileNetv2",
]
