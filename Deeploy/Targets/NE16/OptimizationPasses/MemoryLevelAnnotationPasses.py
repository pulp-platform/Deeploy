# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import SequentialPass
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryLevel


class AnnotateNE16WeightMemoryLevel(SequentialPass):

    def __init__(self, ne16EngineName: str, weightMemoryLevel: MemoryLevel):
        self._weightMemoryLevel = weightMemoryLevel
        self.ne16EngineName = ne16EngineName
        super().__init__()

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:

        def _ne16WeightBufferSize(buffer: ConstantBuffer) -> int:
            return int(np.prod(buffer.shape))  # Weights are encoded as bytes so no need to check for typeWidth

        weightMemoryOccupation = 0

        # Current weight memory occupation
        for buffer in {**ctxt.globalObjects, **ctxt.localObjects}.values():
            if hasattr(buffer, "_memoryLevel") and buffer._memoryLevel == self._weightMemoryLevel.name:
                weightMemoryOccupation += _ne16WeightBufferSize(buffer)

        ne16Nodes = [node for node in graph.nodes if node.attrs["engine"] == self.ne16EngineName]
        for node in ne16Nodes:
            if node.op in ["Conv", "RequantizedConv"]:

                if not (ctxt.is_local(node.inputs[1].name) or ctxt.is_global(node.inputs[1].name)):
                    continue

                buffer = ctxt.lookup(node.inputs[1].name)
                if weightMemoryOccupation + _ne16WeightBufferSize(buffer) < self._weightMemoryLevel.size:
                    buffer._memoryLevel = self._weightMemoryLevel.name
                    weightMemoryOccupation += _ne16WeightBufferSize(buffer)
        return ctxt, graph
