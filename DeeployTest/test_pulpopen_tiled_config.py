# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "PULPOpen"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8
DEFAULT_L2 = 1024000
DEFAULT_MEM_ALLOC_STRATEGY = "MiniMalloc"
DEFAULT_SEARCH_STRATEGY = "random-max"

L2_SINGLEBUFFER_KERNELS = {
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/Integer/Softmax/Regular": [800, 500, 300],
    "Kernels/Integer/Concat": [32000, 16000, 8000],
    "Kernels/Integer/MatMul/Batch": [20000],
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/RMSNorm": [2048, 1024, 512],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 4000],
    "Kernels/Integer/Conv/DW_2D_RQ": [2561],
    "Kernels/Integer/Conv/StriddedPadded_2D_RQ": [600],
    "Kernels/Integer/GEMM/Batch_RQ": [20000],
    "Kernels/Integer/Hardswish/Regular_RQ": [750],
}

L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/Integer/Hardswish/Regular": [750],
    "Kernels/Integer/Softmax/Regular": [1600, 1000, 600],
    "Kernels/Integer/Concat": [64000, 32000, 16000],
    "Kernels/Integer/MatMul/Regular": [64000, 32000, 16000],
    "Kernels/Integer/RMSNorm": [4096, 2048, 1024],
    "Kernels/Integer/Conv/Regular_2D_RQ": [8000, 6000, 5000],
    "Kernels/Integer/Conv/DW_2D_RQ": [5121],
    "Kernels/Integer/Hardswish/Regular_RQ": [800],
}

L2_SINGLEBUFFER_MODELS = {
    "Models/CNN_Linear2": [45000, 30000, 15000],
    "Models/miniMobileNet": [60000, 12000, 6000, 3000],
    "Models/miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Kernels/Integer/Attention": [60000, 10000, 5000],
    "Models/MLPerf/KeywordSpotting": [64000],
    "Models/MLPerf/ImageClassification": [64000],
    "Models/MLPerf/AnomalyDetection": [64000],
}

L2_DOUBLEBUFFER_MODELS = {
    "Models/CNN_Linear2": [60000, 45000, 30000],
    "Models/miniMobileNet": [60000, 24000, 12000, 6000],
    "Models/miniMobileNetv2": [60000, 32000, 24000, 16000],
    "Kernels/Integer/Attention": [60000, 20000, 10000, 5000],
    "Models/MLPerf/KeywordSpotting": [128000],
    "Models/MLPerf/ImageClassification": [128000],
    "Models/MLPerf/AnomalyDetection": [128000],
}
