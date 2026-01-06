# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Snitch platform (tiled)."""

# Snitch tiled platform supports gvsoc, banshee, vsim simulators
# Default configuration: 9 cores, L2 default memory level

DEFAULT_NUM_CORES = 9

# L2 single-buffer tests with different L1 sizes
# Format: (test_name, L1_size)
L2_SINGLEBUFFER_KERNELS = [
    ("TestiNoNorm", 5000),
    ("TestiNoNorm", 10000),
    ("TestAdderLarge", 5000),
    ("TestAdderLarge", 10000),
    ("TestiSoftmaxLarge", 5000),
    ("TestiSoftmaxLarge", 10000),
    ("testRQGEMM", 2000),
    ("testRQGEMM", 5000),
    ("testFloatSoftmax", 2000),
    ("testFloatSoftmax", 5000),
    ("testFloatSoftmax", 10000),
    ("TestRQAdd", 5000),
    ("TestRQAdd", 10000),
    ("testFloatGEMM", 2000),
    ("testFloatGEMM", 5000),
    ("testFloatGEMM", 10000),
    ("testFloatGEMMtransB", 2000),
    ("testFloatGEMMtransB", 5000),
    ("testFloatGEMMtransB", 10000),
]

L2_SINGLEBUFFER_MODELS = []

# Currently no double-buffer configurations in CI
L2_DOUBLEBUFFER_KERNELS = []
L2_DOUBLEBUFFER_MODELS = []
