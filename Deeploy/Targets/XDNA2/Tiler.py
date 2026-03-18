# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

"""XDNA2 tiling constraints and tiling-ready node bindings for MLIR code generation."""

from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.XDNA2.Bindings import XDNA2AddBindings
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

# For Add operator, reuse the generic BOP (Binary Operator) tile constraint
# which handles equal-dimension binary operations
XDNA2AddTilingReadyBindings = TilingReadyNodeBindings(
    nodeBindings=XDNA2AddBindings,
    tileConstraint=AddTileConstraint()
)
