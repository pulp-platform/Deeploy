# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for Cortex-M (QEMU-ARM) platform."""

KERNEL_TESTS = [
    "Adder",
    "MultIO",
    "test1DPad",
    "test2DPad",
    "testMatMul",
    "testMatMulAdd",
    "testMaxPool",
    "testRQConv",
    "testReduceSum",
    "testReduceMean",
    "testSlice",
]

MODEL_TESTS = [
    "simpleRegression",
    "WaveFormer",
]
