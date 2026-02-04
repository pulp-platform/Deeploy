# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.Targets.Generic.TileConstraints.AddTileConstraint import AddTileConstraint
from Deeploy.Targets.Generic.TileConstraints.ConcatTileConstraint import ConcatTileConstraint
from Deeploy.Targets.Generic.TileConstraints.iHardswishTileConstraint import iHardswishTileConstraint
from Deeploy.Targets.Generic.TileConstraints.iRMSNormTileConstraint import iRMSNormTileConstraint
from Deeploy.Targets.Generic.TileConstraints.TransposeTileConstraint import TransposeTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.GatherTileConstraint import GatherTileConstraint
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.Snitch.Bindings import SnitchAddBindings
from Deeploy.Targets.Snitch.Bindings import SnitchConcatBindings
from Deeploy.Targets.Snitch.Bindings import SnitchDivBindings
from Deeploy.Targets.Snitch.Bindings import SnitchGatherBindings
from Deeploy.Targets.Snitch.Bindings import SnitchGemmBindings
from Deeploy.Targets.Snitch.Bindings import SnitchHardSwishBindings
from Deeploy.Targets.Snitch.Bindings import SnitchiNoNormBindings
from Deeploy.Targets.Snitch.Bindings import SnitchiSoftmaxBindings
from Deeploy.Targets.Snitch.Bindings import SnitchMatMulBindings
from Deeploy.Targets.Snitch.Bindings import SnitchMulBindings
from Deeploy.Targets.Snitch.Bindings import SnitchReshapeBindings
from Deeploy.Targets.Snitch.Bindings import SnitchRMSNormBindings
from Deeploy.Targets.Snitch.Bindings import SnitchRQAddBindings
from Deeploy.Targets.Snitch.Bindings import SnitchRqGemmBindings
from Deeploy.Targets.Snitch.Bindings import SnitchTransposeBindings
from Deeploy.Targets.Snitch.TileConstraints import iNoNormTileConstraint
from Deeploy.Targets.Snitch.TileConstraints import iSoftmaxTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.FloatDivTileConstraint import FloatDivTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.FloatMulTileConstraint import FloatMulTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.GemmTileConstraint import GemmTileConstraint
from Deeploy.Targets.Snitch.TileConstraints.ReshapeTileConstraint import ReshapeTileConstraint
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

SnitchRMSNormTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchRMSNormBindings,
                                                           tileConstraint = iRMSNormTileConstraint())

SnitchHardSwishTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchHardSwishBindings,
                                                             tileConstraint = iHardswishTileConstraint())

SnitchDivTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchDivBindings,
                                                       tileConstraint = FloatDivTileConstraint())

SnitchMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchMulBindings,
                                                       tileConstraint = FloatMulTileConstraint())

SnitchMatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchMatMulBindings,
                                                          tileConstraint = MatMulTileConstraint())

SnitchConcatTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchConcatBindings,
                                                          tileConstraint = ConcatTileConstraint())

SnitchTransposeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchTransposeBindings,
                                                             tileConstraint = TransposeTileConstraint())

SnitchReshapeTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchReshapeBindings,
                                                           tileConstraint = ReshapeTileConstraint())

SnitchGatherTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SnitchGatherBindings,
                                                          tileConstraint = GatherTileConstraint())
