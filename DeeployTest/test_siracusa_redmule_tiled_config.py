# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Siracusa platform with RedMulE accelerator (tiled)."""

# Siracusa + RedMulE platform with tiling support
# Default configuration: 8 cores, gvsoc simulator

DEFAULT_CORES = 8

# L2 single-buffer kernel tests
# Format: dict of {test_name: [L1_sizes]}
L2_SINGLEBUFFER_KERNELS = {
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/GEMM/TransB": [8000],
}

# L2 double-buffer kernel tests
L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/FP32/GEMM/Regular": [8000],
}
