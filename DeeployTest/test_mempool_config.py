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
    "Kernels/Integer/Add/MultIO",
    "Kernels/Integer/Add/Regular",
    "Kernels/Integer/Conv/DW_1D",
    "Kernels/Integer/Conv/Regular_1D",
    "Kernels/Integer/Conv/DW_2D",
    "Kernels/Integer/Conv/Regular_2D",
    "Kernels/Integer/GEMM/Regular",
    "Kernels/Integer/MatMul/Add",
    "Kernels/Integer/MatMul/Regular",
    "Kernels/Integer/MaxPool/Regular_2D",
    "Kernels/Integer/Pad/Regular_1D",
    "Kernels/Integer/Pad/Regular_2D",
    "Kernels/Integer/ReduceMean",
    "Kernels/Integer/ReduceSum",
    "Kernels/Integer/Slice",
    "Kernels/Integer/Conv/Regular_2D_RQ",
    "Kernels/Integer/Conv/DW_2D_RQ",
    "Kernels/Integer/GEMM/Regular_RQPerRow",
    "Kernels/Integer/MatMul/Regular_RQ",
]

# Model tests (full networks)
MODEL_TESTS = [
    "Models/CCT/Int/ICCT",
    "Models/CCT/Int/ICCT_8",
    "Models/CCT/Int/ICCT_ITA",
]
