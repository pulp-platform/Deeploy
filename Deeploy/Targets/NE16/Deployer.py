# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Targets.GAP9.Deployer import GAP9Deployer
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import TransposeConstOptPass, TransposeMergePass, \
    TransposeNoPermOptPass, TransposeSplitPass
from Deeploy.Targets.NE16.TopologyOptimizationPasses.Passes import ConvEngineDiscolorationPass, NE16OptimizationPass


class NE16Deployer(GAP9Deployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 default_channels_first = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets = {}):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir, inputOffsets)

        # Keep the global PULPNCHWtoNHWCPass for DW convs (cluster-compatible NHWC).
        # NE16-colored DW convs are fixed up to NE16 NHWC layout inside
        # NE16OptimizationPass below. This avoids breaking cluster-fallback DW convs
        # (stride-2 layers) when --enable-3x3 is on for mixed-engine graphs.

        self.loweringOptimizer.passes += [
            ConvEngineDiscolorationPass(),
            NE16OptimizationPass(self.default_channels_first, "NE16"),
            # NE16OptimizationPass appends its own layout transposes (see
            # _appendTranspose in the NE16 passes). It runs *after* the
            # PULPOpen deployer's transpose clean-up chain, so without
            # re-running that chain here those transposes survive to codegen:
            # consecutive NE16 convs end up separated by a HWC->CHW followed by
            # a CHW->HWC pair that is an identity and should cancel.
            TransposeSplitPass(),
            TransposeMergePass(),
            TransposeConstOptPass(),
            TransposeNoPermOptPass(),
        ]
