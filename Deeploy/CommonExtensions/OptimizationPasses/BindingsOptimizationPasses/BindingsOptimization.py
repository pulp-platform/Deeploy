# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
