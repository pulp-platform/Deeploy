from Deeploy.Targets.Spatz.Bindings import SpatzMatMulBindings, SpatzGatherBindings, SpatzTopKBindings, SpatzSoftmaxBindings
from Deeploy.TilingExtension.TilerExtension import TilingReadyNodeBindings
from Deeploy.Targets.Spatz.TileConstraints.MatMulTileConstraint import MatMulTileConstraint
from Deeploy.Targets.Spatz.TileConstraints.GatherTileConstraint import GatherTileConstraint
from Deeploy.Targets.Spatz.TileConstraints.TopKTileConstraint import TopKTileConstraint
from Deeploy.Targets.Spatz.TileConstraints.SoftmaxTileConstraint import SoftmaxTileConstraint

SpatzMatMulTilingBindings = TilingReadyNodeBindings(nodeBindings = SpatzMatMulBindings,
                                                     tileConstraint = MatMulTileConstraint())

SpatzGatherTilingBindings  = TilingReadyNodeBindings(nodeBindings = SpatzGatherBindings,
                                                     tileConstraint = GatherTileConstraint())

SpatzTopKTilingBindings = TilingReadyNodeBindings(nodeBindings = SpatzTopKBindings,
                                                     tileConstraint = TopKTileConstraint())

SpatzSoftmaxTilingBindings = TilingReadyNodeBindings(nodeBindings = SpatzSoftmaxBindings,
                                                     tileConstraint = SoftmaxTileConstraint())
