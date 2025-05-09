# ----------------------------------------------------------------------
#
# File: RedMulePasses.py
#
# Last edited: 09.05.2025
#
# Copyright (C) 2025, ETH Zurich and University of Bologna.
#
# Author: [Your Name]
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

from functools import partial
import numpy as np
import numpy.typing as npt
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _redmule_weight_layout_fun(graph: gs.Graph, match: Match, name: str):
    """Convert Conv weights from [cout, h, w, cin] to [h,w,cin, cout] for RedMule accelerator"""
    node = list(match.nodes_map.values())[0]
    
    weightTensor = node.inputs[1]
    if isinstance(weightTensor, gs.Constant):
        weightTensor.values = np.transpose(weightTensor.values, (1, 2, 3, 0))
        
    return graph

@contextagnostic
class RedMuleAdjustWeightMemoryLayoutPass(ReplaceSequentialPatternPass):
    """Pass to convert Conv weights from [cout, h, w, cin] to [hwcin, cout] for RedMule accelerator"""

    def __init__(self, redmuleEngineName: str):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['convOut'], op = 'Conv', name = 'conv')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        super().__init__(
            graph,
            _redmule_weight_layout_fun, 
            "_REDMULE_ADJUST_WEIGHT_MEMORY_LAYOUT_PASS")