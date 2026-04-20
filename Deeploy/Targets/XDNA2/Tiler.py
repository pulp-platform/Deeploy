# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 tiling constraints and tiling-ready node bindings for MLIR code generation."""

from Deeploy.Targets.XDNA2.Bindings import XDNA2AddBindings, XDNA2LayerNormBindings, XDNA2SiLUBindings
from Deeploy.Targets.XDNA2.TileConstraints.AddTileConstraint import XDNA2AddTileConstraint
from Deeploy.Targets.XDNA2.TileConstraints.LayerNormTileConstraint import XDNA2LayerNormTileConstraint
from Deeploy.Targets.XDNA2.TileConstraints.UnaryTileConstraint import XDNA2UnaryTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

XDNA2AddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2AddBindings,
                                                      tileConstraint = XDNA2AddTileConstraint())
XDNA2SiLUTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2SiLUBindings,
                                                       tileConstraint = XDNA2UnaryTileConstraint())
XDNA2LayerNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = XDNA2LayerNormBindings,
                                                            tileConstraint = XDNA2LayerNormTileConstraint())
