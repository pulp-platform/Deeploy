# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicConcatBindings, BasicReshapeBindings, \
    BasicTransposeBindings
from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.Generic.TileConstraints.ConcatTileConstraint import ConcatTileConstraint
from Deeploy.Targets.Generic.TileConstraints.NOPTileConstraint import NOPTileConstraint
from Deeploy.Targets.Generic.TileConstraints.TransposeTileConstraint import TransposeTileConstraint
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

BasicTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicTransposeBindings,
                                                            tileConstraint = TransposeTileConstraint())

BasicFlattenTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicReshapeBindings,
                                                          tileConstraint = NOPTileConstraint())

BasicAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = BasicAddBindings,
                                                      tileConstraint = AddTileConstraint())

BasicConcatTilingReadyBinding = TilingReadyNodeBindings(nodeBindings = BasicConcatBindings,
                                                        tileConstraint = ConcatTileConstraint())
