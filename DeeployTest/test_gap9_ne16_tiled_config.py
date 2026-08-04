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
    # the runtime allocator overhead do not fit. The old ceiling was 90KB
    # (">=100KB fails with 'Allocation failed for allocator 2'"); most of that
    # overhead was the cluster slave stacks, which used to be pinned at 3800 B
    # per core with no way to override them. With `-D SLAVESTACKSIZE=1280` the
    # ceiling measured on gvsoc moves to 110KB (121KB and 128KB still fail),
    # and VisualWakeWords goes 860,577 -> 830,146 cycles (8.70 -> 9.02
    # MAC/cycle), still bit-exact. The exact ceiling in 110-121KB is unbisected.
    "Models/MLPerf/VisualWakeWords": [100000],
}

L3_SINGLEBUFFER_MODELS = {}
L3_DOUBLEBUFFER_MODELS = {}
L2_SINGLEBUFFER_KERNELS_WMEM = {}
L3_DOUBLEBUFFER_MODELS_WMEM = {}
