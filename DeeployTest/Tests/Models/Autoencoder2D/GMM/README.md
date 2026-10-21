<!-- SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

This is the combined autoencoder and GMM regression fixture: 43 nodes, input
`x` of shape `[1, 1, 10, 16]`, reconstruction output of the same shape, and
`gmm_output` of shape `[1]`.

It exercises the encoder, decoder and GMM together. The smaller `../../GMM`
fixture isolates the GMM operations; it is not a copy of this model.
