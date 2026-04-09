# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 tiling constraints and tiling-ready node bindings for MLIR code generation."""

from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.Generic.TileConstraints.UnaryTileConstraint import UnaryTileConstraint
from Deeploy.Targets.XDNA2.Bindings import XDNA2AddBindings, XDNA2LayerNormBindings, XDNA2SiLUBindings
from Deeploy.Targets.XDNA2.TileConstraints.LayerNormTileConstraint import XDNA2LayerNormTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

XDNA2AddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2AddBindings,
                                                      tileConstraint = AddTileConstraint())
XDNA2SiLUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2SiLUBindings,
                                                       tileConstraint = UnaryTileConstraint())
XDNA2LayerNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2LayerNormBindings,
                                                            tileConstraint = XDNA2LayerNormTileConstraint())
