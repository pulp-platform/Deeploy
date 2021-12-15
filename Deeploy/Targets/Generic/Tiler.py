# ----------------------------------------------------------------------
#
# File: BasicTiler.py
#
# Last edited: 01.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# - Victor Jung, jungvi@iis.ee.ethz.ch, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
