from Deeploy.Targets.Spatz.Bindings import SpatzMatMulBindings
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint

SpatzMatMulTilingReadyBindings = TilingReadyNodeBindings(nodeBindings = SpatzMatMulBindings,
                                                     tileConstraint = MatMulTileConstraint())