# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Test configuration for GAP9 platform with NE16 accelerator (tiled).

NE16-supported convolution kernels verified to dispatch to NE16
(`ne16_nnx_dispatch` appears in generated Network.c) and PASS on
gvsoc gap9.evk:

- PW 1x1 RQ Conv  (PW_2D_RQ/Regular_RQ)
- PW 1x1 Conv     (PW_2D)
- DW 3x3 RQ Conv  (DW_2D_RQ, with --enable-3x3)
- 3x3 strided RQ  (StriddedPadded_2D_RQ) — falls back to cluster
  because stride 2x2 requires --enableStrides which isn't wired into
  the tiled runner today; still PASS via the cluster kernel.
"""

DEFAULT_CORES = 8

L2_SINGLEBUFFER_KERNELS = {
    "Kernels/Integer/Conv/PW_2D_RQ/Regular_RQ": [32000, 16000],
    "Kernels/Integer/Conv/PW_2D": [32000],
    "Kernels/Integer/Conv/DW_2D_RQ": [32000, 16000],
    "Kernels/Integer/Conv/Dense_2D_RQ": [32000],
    "Kernels/Integer/Conv/StriddedPadded_2D_RQ": [32000],
}

L2_DOUBLEBUFFER_KERNELS = {
    "Kernels/Integer/Conv/PW_2D_RQ/Regular_RQ": [32000],
    "Kernels/Integer/Conv/DW_2D_RQ": [32000],
}

L2_SINGLEBUFFER_MODELS = {
    "Models/MLPerf/VisualWakeWords": [128000],
}

L2_DOUBLEBUFFER_MODELS = {
    # L1 budget is below the full 128KB because the double-buffered tiles plus
    # the runtime allocator overhead do not fit. Two things moved this ceiling:
    # SLAVESTACKSIZE became overridable (freeing ~20KB of L1), and the weight
    # tile's NE16-encoded tail is now pinned, so the solver reserves the full
    # 18432 B the DMA actually writes instead of 16384 B. The latter removes
    # slack that was only ever available by overrunning the requant parameters,
    # so the previous 110000 no longer allocates. 100000 is measured to run.
    #   100000 -> 567,140 cycles, bit-exact
    #   110000 -> "Allocation failed for allocator 2"
    "Models/MLPerf/VisualWakeWords": [100000],
}

L3_SINGLEBUFFER_MODELS = {}
L3_DOUBLEBUFFER_MODELS = {}
L2_SINGLEBUFFER_KERNELS_WMEM = {}
L3_DOUBLEBUFFER_MODELS_WMEM = {}
