# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Snitch platform."""

# Snitch platform supports gvsoc, banshee, vsim simulators
# Default configuration: 9 cores

DEFAULT_NUM_CORES = 9

KERNEL_TESTS = [
    "Kernels/FP32/Softmax/Regular",
    "Kernels/Integer/Add/Large",
    "Kernels/Integer/Add/Regular",
    "Kernels/Integer/Softmax/Large",
    "Kernels/Integer/Softmax/Regular",
    "Kernels/Integer/MatMul/Regular",
    "Kernels/Integer/iNoNorm",
    "Kernels/Integer/GEMM/Regular_RQPerRow",
    "Kernels/Integer/Add/Regular_RQ",
    "Kernels/Integer/GEMM/TransB_RQ",
]

MODEL_TESTS = [
    "Models/microLlama/microLlama_fp32_1",
]
