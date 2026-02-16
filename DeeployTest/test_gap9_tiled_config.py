# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "GAP9"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8
DEFAULT_L2 = 1024000
DEFAULT_MEM_ALLOC_STRATEGY = "MiniMalloc"
DEFAULT_SEARCH_STRATEGY = "random-max"

L2_SINGLEBUFFER_KERNELS = {
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 4000],
    "Kernels/Integer/Conv/StriddedPadded_2D_RQ": [600],
    "Kernels/Integer/Conv/DW_2D_RQ": [2561],
    "Kernels/Integer/Softmax/Regular": [800, 500, 300],
    "Kernels/Integer/Concat": [32000, 16000, 8000],
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/Conv/Regular_2D_NoBias": [6600],
    "Kernels/FP32/Conv/Regular_2D_ZeroValuedBias": [6600],
    "Kernels/FP32/Conv/DW_2D_Bias": [7200],
    "Kernels/FP32/Conv/DW_2D_NoBias": [7200],
    "Kernels/FP32/Conv/DW_2D_ZeroValuedBias": [7200],
    "Kernels/FP32/LayerNorm": [2000],
    "Kernels/FP32/MaxPool/Regular_2D": [2000],
    "Kernels/FP32/MatMul": [2000],
    "Kernels/FP32/ReLU": [2000],
    "Kernels/FP32/Reshape/SkipConnection": [1400],
    "Kernels/FP32/Softmax/Regular": [4000],
    "Kernels/FP32/Transpose": [2000],
    "Kernels/FP32/Mul": [2000],
    "Kernels/Integer/GEMM/Batch_RQ": [20000],
    "Kernels/Integer/MatMul/Batch": [20000],
}

L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 5000],
    "Kernels/Integer/Conv/DW_2D_RQ": [5121],
    "Kernels/Integer/Softmax/Regular": [1600, 1000, 600],
    "Kernels/Integer/Concat": [64000, 32000, 16000],
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/FP32/GEMM/Regular": [8000],
    "Kernels/FP32/Conv/Regular_2D_NoBias": [8800],
    "Kernels/FP32/Conv/Regular_2D_ZeroValuedBias": [8800],
    "Kernels/FP32/Conv/DW_2D_Bias": [9800],
    "Kernels/FP32/Conv/DW_2D_NoBias": [10000],
    "Kernels/FP32/Conv/DW_2D_ZeroValuedBias": [9800],
    "Kernels/FP32/LayerNorm": [2000],
    "Kernels/FP32/MaxPool/Regular_2D": [5000],
    "Kernels/FP32/MatMul": [5000],
    "Kernels/FP32/ReLU": [20],
    "Kernels/FP32/Reshape/SkipConnection": [2600],
    "Kernels/FP32/Softmax/Regular": [8000],
    "Kernels/FP32/Transpose": [2000],
    "Kernels/FP32/Mul": [2000],
}

L2_SINGLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 12000, 6000, 3000],
    "Models/miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Models/MLPerf/KeywordSpotting": [64000],
    "Models/MLPerf/ImageClassification": [64000],
    "Models/MLPerf/AnomalyDetection": [64000],
}

L2_DOUBLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 24000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 32000, 24000, 16000],
    "Models/MLPerf/KeywordSpotting": [64000],
    "Models/MLPerf/ImageClassification": [64000],
    "Models/MLPerf/AnomalyDetection": [64000],
}

L3_SINGLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Models/CCT/FP32/CCT_2_32_32_128": [128000],
}

L3_DOUBLEBUFFER_MODELS = {
    "Models/miniMobileNet": [60000, 24000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 32000, 24000, 16000],
}
