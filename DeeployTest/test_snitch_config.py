# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""Test configuration for Snitch platform."""

# Snitch platform supports gvsoc, banshee, vsim simulators
# Default configuration: 9 cores

DEFAULT_NUM_CORES = 9

KERNEL_TESTS = [
    "Adder",
    "iSoftmax",
    "TestiNoNorm",
    "TestAdderLarge",
    "TestiSoftmaxLarge",
    "testMatMul",
    "testRQGEMM",
    "TestRQAdd",
    "testRQGEMMTransB",
    "testFloatSoftmax",
]

MODEL_TESTS = []
