# ----------------------------------------------------------------------
#
# File: MemoryLevelAnnotationPasses.py
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Victor Jung, ETH Zurich
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

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import SequentialPass
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryLevel


class AnnotateNeurekaWeightMemoryLevel(SequentialPass):

    def __init__(self, neurekaEngineName: str, weightMemoryLevel: MemoryLevel):
        self._weightMemoryLevel = weightMemoryLevel
        self.neurekaEngineName = neurekaEngineName
        super().__init__()

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:

        def _neurekaWeightBufferSize(buffer: ConstantBuffer) -> int:
            return int(np.prod(buffer.shape))  # Weights are encoded as bytes so no need to check for typeWidth

        weightMemoryOccupation = 0

        # Current weight memory occupation
        for buffer in {**ctxt.globalObjects, **ctxt.localObjects}.values():
            if hasattr(buffer, "_memoryLevel") and buffer._memoryLevel == self._weightMemoryLevel.name:
                weightMemoryOccupation += _neurekaWeightBufferSize(buffer)

        neurekaNodes = [node for node in graph.nodes if node.attrs["engine"] == self.neurekaEngineName]
        for node in neurekaNodes:
            if node.op in ["Conv", "RequantizedConv"]:

                if not (ctxt.is_local(node.inputs[1].name) or ctxt.is_global(node.inputs[1].name)):
                    continue

                buffer = ctxt.lookup(node.inputs[1].name)
                if weightMemoryOccupation + _neurekaWeightBufferSize(buffer) < self._weightMemoryLevel.size:
                    buffer._memoryLevel = self._weightMemoryLevel.name
                    weightMemoryOccupation += _neurekaWeightBufferSize(buffer)
        return ctxt, graph
