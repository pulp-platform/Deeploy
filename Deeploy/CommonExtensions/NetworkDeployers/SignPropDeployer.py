# ----------------------------------------------------------------------
#
# File: SignPropDeployer.py
#
# Last edited: 11.06.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
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
from Deeploy.DeeployTypes import DeploymentPlatform, NetworkDeployer, TopologyOptimizer


class SignPropDeployer(NetworkDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 inputOffsets: Dict[str, int] = {}):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir)

        if inputOffsets == {}:
            for key in inputTypes.keys():
                inputOffsets[key] = 0

        self.inputOffsets = inputOffsets

    def _createIOBindings(self, ctxt, graph):
        ctxt = super()._createIOBindings(ctxt, graph)
        for node in graph.inputs:
            data_name = node.name
            nb = ctxt.lookup(data_name)
            data_type = self.inputTypes[data_name]
            nb._signed = (self.inputOffsets[data_name] == 0)
            nb.nLevels = (2**data_type.referencedType.typeWidth)

        return ctxt
