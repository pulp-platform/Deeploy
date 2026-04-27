from Deeploy.Targets.Spatz.Bindings import SpatzMatMulBindings, SpatzGatherBindings, SpatzTopKBindings
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings
from Deeploy.Targets.PULPOpen.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.Spatz.TileConstraints.GatherTileConstraint import GatherTileConstraint
from Deeploy.Targets.Spatz.TileConstraints.TopKTileConstraint import TopKTileConstraint

SpatzMatMulTilingBindings = TilingReadyNodeBindings(nodeBindings = SpatzMatMulBindings,
                                                     tileConstraint = MatMulTileConstraint())

SpatzGatherTilingBindings  = TilingReadyNodeBindings(nodeBindings = SpatzGatherBindings,
                                                     tileConstraint = GatherTileConstraint())

SpatzTopKTilingBindings = TilingReadyNodeBindings(nodeBindings = SpatzTopKBindings,
                                                     tileConstraint = TopKTileConstraint())
