# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Siracusa platform with Neureka accelerator (tiled)."""

# Siracusa + Neureka platform with tiling support
# Default configuration: 8 cores, gvsoc simulator

DEFAULT_CORES = 8

# L2 single-buffer kernel tests
# Format: dict of {test_name: [L1_sizes]}
L2_SINGLEBUFFER_KERNELS = {
    "Kernels/Integer/GEMM/Regular_RQPerColumn": [16000],
    "Kernels/Integer/Conv/PW_2D": [32000],
    "Kernels/Integer/Conv/PW_2D_RQ/Regular_RQ": [32000],
    "Kernels/Integer/Conv/PW_2D_RQ/Unsigned_RQ": [32000],
}

# L2 double-buffer kernel tests
L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/Integer/GEMM/Regular_RQPerColumn": [16000],
    "Kernels/Integer/Conv/PW_2D": [32000],
    "Kernels/Integer/Conv/PW_2D_RQ/Regular_RQ": [32000],
    "Kernels/Integer/Conv/PW_2D_RQ/Unsigned_RQ": [32000],
}

# L3 single-buffer model tests
# Format: dict of {test_name: [L1_sizes]}
L3_SINGLEBUFFER_MODELS = {
    "Models/miniMobileNet": [2000],
    "Kernels/Integer/Attention": [2500],
    "Models/Transformer": [15000],
    "Models/microLlama/INT8/microLlama1": [10000],
}

# L3 double-buffer model tests
L3_DOUBLEBUFFER_MODELS = {
    "Models/miniMobileNet": [2000],
    "Kernels/Integer/Attention": [5000],
    "Models/Transformer": [30000],
}

# L2 single-buffer kernel tests with weight memory (neureka-wmem)
L2_SINGLEBUFFER_KERNELS_WMEM = {
    "Kernels/Integer/GEMM/Regular_RQPerColumn": [16000],
    "Kernels/Integer/Conv/PW_2D": [32000],
    "Kernels/Integer/Conv/PW_2D_RQ/Regular_RQ": [32000],
    "Kernels/Integer/Conv/PW_2D_RQ/Unsigned_RQ": [32000],
}

# L3 double-buffer model tests with weight memory (neureka-wmem)
L3_DOUBLEBUFFER_MODELS_WMEM = {
    "Models/miniMobileNet": [2000],
    "Kernels/Integer/Attention": [3500],
    "Models/microLlama/INT8/microLlama1": [10000],
}
