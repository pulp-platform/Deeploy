# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

PLATFORM_NAME = "Siracusa"
SIMULATOR = "gvsoc"
DEFAULT_CORES = 8

# Tiled gradient-operator kernel tests for Siracusa (SBTiler, single backward-pass operators).

L2_SINGLEBUFFER_GRAD_KERNELS = {
    # ── Pointwise / element-wise grad ops ──────────────────────────────────
    "Kernels/FP32/SoftmaxGrad": [2000],
    "Kernels/FP32/ReluGrad": [8000],
    "Kernels/FP32/GeLUGrad": [2000],
    # ── Normalisation grad ops ──────────────────────────────────────────────
    "Kernels/FP32/LayernormGrad": [4000],
    "Kernels/FP32/GroupNormGrad": [8000],
    # ── Pooling grad ops ───────────────────────────────────────────────────
    "Kernels/FP32/MaxpoolGrad": [2000],
    "Kernels/FP32/AveragePoolGrad": [4000],
    # ── Convolution grad ops (small / synthetic) ───────────────────────────
    "Kernels/FP32/ConvGrad": [2000],
    # ── Depthwise convolution grad ops ─────────────────────────────────────
    "Kernels/FP32/DWConvGrad": [8000],
    # ── Pointwise convolution grad ops ─────────────────────────────────────
    "Kernels/FP32/PWConvGrad": [8000]
}
