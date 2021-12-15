# ----------------------------------------------------------------------
#
# File: BindingsOptimization.py
#
# Last edited: 21.11.2023
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

from typing import Dict, Tuple

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, ONNXLayer


class BindingOptimizationPass():

    def apply(self, ctxt: NetworkContext, graph: gs.Graph,
              layerBinding: Dict[str, ONNXLayer]) -> Tuple[NetworkContext, Dict[str, ONNXLayer]]:
        return ctxt, layerBinding


class BindingOptimizer():

    def optimize(self, ctxt: NetworkContext, graph: gs.Graph,
                 layerBinding: Dict[str, ONNXLayer]) -> Tuple[NetworkContext, Dict[str, ONNXLayer]]:
        newLayerBinding = layerBinding.copy()
        for _pass in self.passes:
            ctxt, newLayerBinding = _pass.apply(ctxt, graph, newLayerBinding)
            assert newLayerBinding.keys() == layerBinding.keys(), "BindingOptimizationPass removed bindings!"
        return ctxt, layerBinding
