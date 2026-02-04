# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable, Dict, Type

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    NCHWtoNHWCPass
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RemoveGlobalOutputReshapePass
from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    TransposeMatmulInputsPass
from Deeploy.DeeployTypes import DeploymentPlatform
from Deeploy.DeeployTypes import TopologyOptimizer
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ReshapeConstOptPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import TransposeConstOptPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import TransposeMergePass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import TransposeSplitPass


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
