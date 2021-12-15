# ----------------------------------------------------------------------
#
# File: Deployer.py
#
# Last edited: 26.07.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    NeurekaNCHWtoNHWCPass, PULPNCHWtoNHWCPass
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
                    self.loweringOptimizer.passes[idx] = NeurekaNCHWtoNHWCPass(self.default_channels_first)

        self.loweringOptimizer.passes += [
            ConvEngineDiscolorationPass(),
            NeurekaOptimizationPass(self.default_channels_first, "Neureka")
        ]
