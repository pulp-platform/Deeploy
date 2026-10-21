<!-- SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This is the standalone GMM regression fixture: 8 nodes, input `x` of shape
`[4, 8]`, and one output of shape `[4]`. It isolates the MatMul, Add, Mul and
ReduceLogSumExp operations from the autoencoder.

The separate `../Autoencoder2D/GMM` fixture exercises the complete autoencoder
and GMM integration, including convolution, normalization and reconstruction.
The two fixtures contain different graphs, weights and test data.
