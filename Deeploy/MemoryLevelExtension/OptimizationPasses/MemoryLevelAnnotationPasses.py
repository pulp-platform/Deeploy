# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from typing import List, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import SequentialPass
from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel


class AnnotateDefaultMemoryLevel(SequentialPass):

    def __init__(self, memoryHierarchy: MemoryHierarchy):
        super().__init__()
        self.memoryHierarchy = memoryHierarchy

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        for _buffer in {**ctxt.localObjects, **ctxt.globalObjects}.values():
            if not hasattr(_buffer, "_memoryLevel"):
                _buffer._memoryLevel = self.memoryHierarchy.getDefaultMemoryLevel().name
        return ctxt, graph


class AnnotateIOMemoryLevel(SequentialPass):

    def __init__(self, ioLevel: str):
        super().__init__()
        self.ioLevel = ioLevel

    def apply(self, ctxt: NetworkContext, graph: gs.Graph) -> Tuple[NetworkContext, gs.Graph]:
        buffers = []

        def globalBuffers(tensors: List[gs.Tensor]) -> List[VariableBuffer]:
            return [ctxt.globalObjects[tensor.name] for tensor in tensors if tensor.name in ctxt.globalObjects.keys()]

        inputBuffers = globalBuffers(graph.inputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer) and len(_buffer._users) > 0,
                          inputBuffers)

        outputBuffers = globalBuffers(graph.outputs)
        buffers += filter(lambda _buffer: isinstance(_buffer, ctxt.VariableBuffer), outputBuffers)

        for _buffer in buffers:
            _buffer._memoryLevel = self.ioLevel

        return ctxt, graph


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
