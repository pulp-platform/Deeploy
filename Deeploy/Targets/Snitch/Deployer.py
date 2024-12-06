# ----------------------------------------------------------------------
#
# File: SnitchDeployer.py
#
# Last edited: 23.04.2024
#
# Copyright (C) 2024, ETH Zurich and University of Bologna.
#
# Authors:
# - Philip Wiese (wiesep@iis.ee.ethz.ch), ETH Zurich
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
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    NCHWtoNHWCPass, RemoveGlobalOutputReshapePass, TransposeMatmulInputsPass
from Deeploy.DeeployTypes import DeploymentPlatform, TopologyOptimizer
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ReshapeConstOptPass, TransposeConstOptPass, \
    TransposeMergePass, TransposeSplitPass


class SnitchDeployer(SignPropDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first = False,
                 deeployStateDir: str = "DeeployStateDir",
                 inputOffsets = {}):
        super().__init__(graph,
                         deploymentPlatform,
                         inputTypes,
                         loweringOptimizer,
                         scheduler,
                         name,
                         default_channels_first = default_channels_first,
                         deeployStateDir = deeployStateDir,
                         inputOffsets = inputOffsets)

        self.loweringOptimizer.passes += [
            TransposeMatmulInputsPass(),
            NCHWtoNHWCPass(self.default_channels_first),
            TransposeSplitPass(),
            TransposeMergePass(),
            TransposeConstOptPass(),
            ReshapeConstOptPass(),
            RemoveGlobalOutputReshapePass(),
        ]
