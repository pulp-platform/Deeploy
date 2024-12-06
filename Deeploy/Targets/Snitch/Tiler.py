# ----------------------------------------------------------------------
#
# File: SnitchTiler.py
#
# Last edited: 03.06.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
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

from Deeploy.Targets.Snitch.Bindings import SnitchAddBindings, SnitchGemmBindings, SnitchiNoNormBindings, \
    SnitchiSoftmaxBindings, SnitchRQAddBindings, SnitchRqGemmBindings
from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.Snitch.TileConstraints import iNoNormTileConstraint, iSoftmaxTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.GemmTileConstraint import GemmTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.RqGemmTileConstraint import RqGemmTileConstraint

from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings

SnitchiSoftmaxTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchiSoftmaxBindings,
                                                            tileConstraint = iSoftmaxTileConstraint())
SnitchiNoNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchiNoNormBindings,
                                                           tileConstraint = iNoNormTileConstraint())
SnitchRQAddTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchRQAddBindings,
                                                         tileConstraint = AddTileConstraint())
SnitchGemmTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchGemmBindings,
                                                        tileConstraint = GemmTileConstraint())
SnitchRqGemmTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchRqGemmBindings,
                                                          tileConstraint = RqGemmTileConstraint())

SnitchAddTileReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchAddBindings,
                                                     tileConstraint = AddTileConstraint())
