# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "Siracusa"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8
DEFAULT_L2 = 1024000
DEFAULT_MEM_ALLOC_STRATEGY = "MiniMalloc"
DEFAULT_SEARCH_STRATEGY = "random-max"

L2_SINGLEBUFFER_KERNELS = {
    "testMatMul": [64000, 32000, 16000],
    "test2DRequantizedConv": [8000, 6000, 4000],
    "test2DRequantizedStriddedPaddedConv": [600],
    "testRequantizedDWConv": [2561],
    "iSoftmax": [800, 500, 300],
    "testConcat": [32000, 16000, 8000],
    "testRMSNorm": [2048, 1024, 512],
    "Hardswish": [750],
    "RQHardswish": [750],
    "testFloatGEMM": [8000],
    "testFloat2DConvolution": [1600],
    "testFloat2DConvolutionBias": [6600],
    "testFloat2DConvolutionZeroBias": [6600],
    "testFloat2DDWConvolution": [7200],
    "testFloat2DDWConvolutionBias": [7200],
    "testFloat2DDWConvolutionZeroBias": [7200],
    "testFloatLayerNorm": [2000],
    "testFloatMaxPool": [2000],
    "testFloatMatmul": [2000],
    "testFloatRelu": [2000],
    "testFloatReshapeWithSkipConnection": [1400],
    "testFloatSoftmax": [4000],
    "testFloatTranspose": [2000],
    "testFloatMul": [2000],
    "largeFloatAdd": [220000],
    "testRQGEMMwBatch": [20000],
    "testMatMulBatch": [20000],
}

L2_DOUBLEBUFFER_KERNELS = {
    "testMatMul": [64000, 32000, 16000],
    "test2DRequantizedConv": [8000, 6000, 5000],
    "testRequantizedDWConv": [5121],
    "iSoftmax": [1600, 1000, 600],
    "testConcat": [64000, 32000, 16000],
    "testRMSNorm": [4096, 2048, 1024],
    "Hardswish": [750],
    "RQHardswish": [800],
    "testFloatGEMM": [8000],
    "testFloat2DConvolution": [2000],
    "testFloat2DConvolutionBias": [8800],
    "testFloat2DConvolutionZeroBias": [8800],
    "testFloat2DDWConvolution": [9800],
    "testFloat2DDWConvolutionBias": [10000],
    "testFloat2DDWConvolutionZeroBias": [9800],
    "testFloatLayerNorm": [2000],
    "testFloatMaxPool": [5000],
    "testFloatMatmul": [5000],
    "testFloatRelu": [20],
    "testFloatReshapeWithSkipConnection": [2600],
    "testFloatSoftmax": [8000],
    "testFloatTranspose": [2000],
    "testFloatMul": [2000],
}

L2_SINGLEBUFFER_MODELS = {
    "simpleRegression": [45000, 30000, 15000],
    "miniMobileNet": [60000, 12000, 6000, 3000],
    "miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Attention": [60000, 10000, 5000],
    "microLlama/microLlama1": [60000, 10000, 5000],
    "microLlama/microLlama8": [60000, 10000, 5000],
    "microLlama/microLlama8_parallel": [60000, 10000, 5000],
    "MLPerf/KeywordSpotting": [64000],
    "MLPerf/ImageClassification": [64000],
    "MLPerf/AnomalyDetection": [64000],
    "CCT/CCT_1_16_16_8": [64000],
    "testFloatDemoTinyViT": [4000],
}

L3_SINGLEBUFFER_MODELS = {
    "simpleRegression": [45000, 30000, 16000],
    "miniMobileNet": [60000, 12000, 6000],
    "miniMobileNetv2": [60000, 16000, 12000, 8000],
    "Attention": [60000, 10000, 5000, 2500],
    "Transformer": [60000, 30000, 15000],
    "microLlama/microLlama1": [60000, 10000, 5000],
    "CCT/CCT_2_32_32_128": [128000],
    "testTrainCCT/CCT2_FT2": [128000],
    "testFloatDemoTinyViT": [4000],
}

L3_DOUBLEBUFFER_MODELS = {
    "simpleRegression": [60000, 45000, 30000],
    "miniMobileNet": [60000, 24000, 12000, 6000],
    "miniMobileNetv2": [60000, 32000, 24000, 16000],
    "Attention": [60000, 20000, 10000, 5000],
    "Transformer": [60000, 30000, 15000],
    "microLlama/microLlama1": [60000, 20000, 10000],
    "microLlama/microLlama8": [60000, 20000, 10000],
    "microLlama/microLlama8_parallel": [60000, 20000, 10000],
    "CCT/CCT_2_32_32_128": [128000],
    "testTrainCCT/CCT2_FT2": [128000],
    "testFloatDemoTinyViT": [4000],
}
