# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    NCHWtoNHWCPass, PULPNCHWtoNHWCPass
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Targets.Neureka.TopologyOptimizationPasses.Passes import ConvEngineDiscolorationPass, \
    NeurekaOptimizationPass
from Deeploy.Targets.PULPOpen.Deployer import PULPDeployer


class NeurekaDeployer(PULPDeployer):

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

        if self.Platform.engines[0].enable3x3:
            for idx in range(len(self.loweringOptimizer.passes)):
                if isinstance(self.loweringOptimizer.passes[idx], PULPNCHWtoNHWCPass):
                    self.loweringOptimizer.passes[idx] = NCHWtoNHWCPass(self.default_channels_first)

        self.loweringOptimizer.passes += [
            ConvEngineDiscolorationPass(),
            NeurekaOptimizationPass(self.default_channels_first, "Neureka")
        ]
