# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Cortex-M (QEMU-ARM) platform."""

KERNEL_TESTS = [
    "Kernels/Integer/Add/Regular",
    "Kernels/Integer/Add/MultIO",
    "Kernels/Integer/Pad/Regular_1D",
    "Kernels/Integer/Pad/Regular_2D",
    "Kernels/Integer/MatMul/Regular",
    "Kernels/Integer/MatMul/Add",
    "Kernels/Integer/MaxPool/Regular_2D",
    "Kernels/Integer/Conv/Regular_2D_RQ",
    "Kernels/Integer/ReduceSum",
    "Kernels/Integer/ReduceMean",
    "Kernels/Integer/Slice",
]

MODEL_TESTS = [
    "Models/CNN_Linear2",
    "Models/WaveFormer",
]
