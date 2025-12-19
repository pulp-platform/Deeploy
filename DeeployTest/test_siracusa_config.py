# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "Siracusa"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8

KERNEL_TESTS = [
    "Adder",
    "MultIO",
    "test1DPad",
    "test2DPad",
    "testMatMul",
    "testMatMulAdd",
    "testRequantizedDWConv",
    "test2DRequantizedConv",
    "iSoftmax",
    "testConcat",
    "testRMSNorm",
    "trueIntegerDivSandwich",
    "Hardswish",
    "RQHardswish",
    "testBacktracking",
    "testFloatAdder",
    "testFloatGEMM",
    "testFloat2DConvolution",
    "testFloat2DConvolutionBias",
    "testFloat2DConvolutionZeroBias",
    "testFloat2DDWConvolution",
    "testFloat2DDWConvolutionBias",
    "testFloat2DDWConvolutionZeroBias",
    "testFloatLayerNorm",
    "testFloatRelu",
    "testFloatMaxPool",
    "testFloatMatmul",
    "testFloatSoftmax",
    "testFloatTranspose",
    "testFloatMul",
    "Quant",
    "Dequant",
    "testFloatReduceSum",
    "testFloatReshapeWithSkipConnection",
    "testFloatSoftmaxGrad",
    "testFloatSoftmaxCrossEntropy",
    "testFloatSoftmaxCrossEntropyGrad",
    "QuantizedLinear",
]

MODEL_TESTS = [
    "simpleRegression",
    "miniMobileNet",
    "miniMobileNetv2",
    "Attention",
    "MLPerf/KeywordSpotting",
    "MLPerf/ImageClassification",
    "MLPerf/AnomalyDetection",
    "CCT/CCT_1_16_16_8",
    "CCT/CCT_2_32_32_128_Opset20",
    "testFloatDemoTinyViT",
]
