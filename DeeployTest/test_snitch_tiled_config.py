# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Snitch platform (tiled)."""

# Snitch tiled platform supports gvsoc, banshee, vsim simulators
# Default configuration: 9 cores, L2 default memory level

DEFAULT_NUM_CORES = 9

# L2 single-buffer tests with different L1 sizes
# Format: {test_name: [L1_sizes]}
L2_SINGLEBUFFER_KERNELS = {
    "Kernels/Integer/Add/Large": [5000, 10000],
    "Kernels/Integer/Softmax/Large": [5000, 10000],
    "Kernels/FP32/Softmax/Regular": [2000, 5000, 10000],
    "Kernels/FP32/GEMM/Regular": [2000, 5000, 10000],
    "Kernels/FP32/GEMM/TransB": [2000, 5000, 10000],
    "Kernels/Integer/iNoNorm": [5000, 10000],
    "Kernels/Integer/Add/Regular_RQ": [5000, 10000],
    "Kernels/Integer/GEMM/Regular_RQPerRow": [2000, 5000],
}

L2_SINGLEBUFFER_MODELS = {
    "Models/microLlama/microLlama_fp32_1": [10000, 20000],
}

# Currently no double-buffer configurations in CI
L2_DOUBLEBUFFER_KERNELS = {}
L2_DOUBLEBUFFER_MODELS = {}
