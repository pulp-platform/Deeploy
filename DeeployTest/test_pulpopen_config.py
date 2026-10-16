# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "PULPOpen"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8

KERNEL_TESTS = [
    "Kernels/Integer/Hardswish/Regular",
    "Kernels/Integer/Softmax/Regular",
    "Kernels/Integer/Add/MultIO",
    "Kernels/Integer/Add/Regular",
    "Kernels/Integer/Concat",
    "Kernels/Integer/MatMul/Add",
    "Kernels/Integer/MatMul/Regular",
    "Kernels/Integer/Pad/Regular_1D",
    "Kernels/Integer/Pad/Regular_2D",
    "Kernels/Integer/RMSNorm",
    "Kernels/Integer/Conv/Regular_1D_RQ",
    "Kernels/Integer/Conv/Regular_2D_RQ",
    "Kernels/Integer/Conv/DW_2D_RQ",
    "Kernels/Integer/Hardswish/Regular_RQ",
    "Kernels/Integer/TrueIntegerDiv",
    "Others/Backtracking",
]

MODEL_TESTS = [
    "Kernels/Integer/Attention",
    "Models/CNN_Linear1",
    "Models/CNN_Linear2",
    "Models/miniMobileNet",
    "Models/miniMobileNetv2",
    "Models/MLPerf/KeywordSpotting",
    "Models/MLPerf/ImageClassification",
    "Models/MLPerf/AnomalyDetection",
]