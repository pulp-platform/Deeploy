# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for Siracusa platform with Neureka accelerator (tiled)."""

# Siracusa + Neureka platform with tiling support
# Default configuration: 8 cores, gvsoc simulator

DEFAULT_CORES = 8

# L2 single-buffer kernel tests
# Format: dict of {test_name: [L1_sizes]}
L2_SINGLEBUFFER_KERNELS = {
    "testRequantizedLinear": [16000],
    "testPointwise": [32000],
    "testPointwiseConvBNReLU": [32000],
    "testPointwiseUnsignedWeights": [32000],
}

# L2 double-buffer kernel tests
L2_DOUBLEBUFFER_KERNELS = {
    "testRequantizedLinear": [16000],
    "testPointwise": [32000],
    "testPointwiseConvBNReLU": [32000],
    "testPointwiseUnsignedWeights": [32000],
}

# L3 single-buffer model tests
# Format: dict of {test_name: [L1_sizes]}
L3_SINGLEBUFFER_MODELS = {
    "miniMobileNet": [2000],
    "Attention": [2500],
    "Transformer": [15000],
    "microLlama/microLlama1": [10000],
}

# L3 double-buffer model tests
L3_DOUBLEBUFFER_MODELS = {
    "miniMobileNet": [2000],
    "Attention": [5000],
    "Transformer": [30000],
}

# L2 single-buffer kernel tests with weight memory (neureka-wmem)
L2_SINGLEBUFFER_KERNELS_WMEM = {
    "testRequantizedLinear": [16000],
    "testPointwise": [32000],
    "testPointwiseConvBNReLU": [32000],
    "testPointwiseUnsignedWeights": [32000],
}

# L3 double-buffer model tests with weight memory (neureka-wmem)
L3_DOUBLEBUFFER_MODELS_WMEM = {
    "miniMobileNet": [2000],
    "Attention": [3500],
    "microLlama/microLlama1": [10000],
}
