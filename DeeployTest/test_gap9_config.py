# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for GAP9 platform."""

DEFAULT_NUM_CORES = 8

KERNEL_TESTS = [
    "Kernels/Integer/Add/Regular", "Kernels/Integer/Add/MultIO", "Kernels/Integer/Pad/Regular_1D",
    "Kernels/Integer/Pad/Regular_2D", "Kernels/Integer/MatMul/Regular", "Kernels/Integer/MatMul/Add",
    "Kernels/Integer/Conv/DW_2D_RQ", "Kernels/Integer/Conv/Regular_2D_RQ", "Kernels/Integer/Softmax/Regular",
    "Kernels/Integer/Concat", "Kernels/Integer/Hardswish/Regular", "Others/Backtracking", "Kernels/FP32/Add/Regular",
    "Kernels/FP32/GEMM/Regular", "Kernels/FP32/Conv/Regular_2D_Bias", "Kernels/FP32/Conv/Regular_2D_NoBias",
    "Kernels/FP32/Conv/Regular_2D_ZeroValuedBias", "Kernels/FP32/Conv/DW_2D_Bias", "Kernels/FP32/Conv/DW_2D_NoBias",
    "Kernels/FP32/Conv/DW_2D_ZeroValuedBias", "Kernels/FP32/LayerNorm", "Kernels/FP32/ReLU",
    "Kernels/FP32/MaxPool/Regular_2D", "Kernels/FP32/MatMul", "Kernels/FP32/Softmax/Regular", "Kernels/FP32/Transpose",
    "Kernels/FP32/Mul", "Kernels/Mixed/Dequant", "Kernels/Mixed/Quant", "Kernels/FP32/ReduceSum",
    "Kernels/FP32/Reshape/SkipConnection"
]

MODEL_TESTS = [
    "Models/miniMobileNet",
    "Models/miniMobileNetv2",
    "Models/MLPerf/KeywordSpotting",
    "Models/MLPerf/ImageClassification",
    "Models/MLPerf/AnomalyDetection",
]
