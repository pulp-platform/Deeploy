# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Runtime-sequence pass that configures shim DMA for L3 ↔ L1 transfers.

Given an :class:`MLIRExecutionBlock` whose device-phase passes have already
populated ``fifoMap``, ``numElements``, and ``runtimeSequenceArgs``, this
pass emits ``aiex_d.dma_configure_task_for`` / ``dma_start_task`` /
``dma_await_task`` / ``dma_free_task`` operations directly into the current
``@aiex_d.runtime_sequence`` insertion point.

The pass is operator-agnostic — it iterates over the FIFO map and
runtime-sequence arguments to configure DMA for every input and output
tensor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import aie.ir as ir
from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRRuntimeSequencePass(MLIRCodeTransformationPass):
    """Emit DMA configuration inside a ``runtime_sequence`` block.

    Parameters
    ----------
    inputTensorKeys : list of str
        Keys in ``operatorRepresentation`` that name input tensors.
    outputTensorKeys : list of str
        Keys that name output tensors.
    """

    def __init__(self, inputTensorKeys: list, outputTensorKeys: list) -> None:
        self.inputTensorKeys = inputTensorKeys
        self.outputTensorKeys = outputTensorKeys

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        numElements = mlirBlock.numElements
        seqArgs = mlirBlock.runtimeSequenceArgs

        dims = [
            aie_d.bd_dim_layout(size = 1, stride = 0),
            aie_d.bd_dim_layout(size = 1, stride = 0),
            aie_d.bd_dim_layout(size = 1, stride = 0),
            aie_d.bd_dim_layout(size = numElements, stride = 1),
        ]

        # Build ordered list of (fifoName, seqArg, isOutput)
        transfers = []
        allKeys = self.inputTensorKeys + self.outputTensorKeys
        for idx, key in enumerate(allKeys):
            fifoName = mlirBlock.fifoMap[key]
            isOutput = key in self.outputTensorKeys
            transfers.append((fifoName, seqArgs[idx], isOutput))

        inputTasks = []
        outputTasks = []

        for fifoName, seqArg, isOutput in transfers:
            if isOutput:
                task = aiex_d.dma_configure_task_for(fifoName, issue_token = True)
            else:
                task = aiex_d.dma_configure_task_for(fifoName)
            block = task.body.blocks.append()
            with ir.InsertionPoint(block):
                aie_d.dma_bd(seqArg, offset = 0, len = numElements, dimensions = dims, burst_length = 0)
                aie_d.end()
            aiex_d.dma_start_task(task)

            if isOutput:
                outputTasks.append(task)
            else:
                inputTasks.append(task)

        # Await output tasks, then free input tasks
        for task in outputTasks:
            aiex_d.dma_await_task(task)
        for task in inputTasks + outputTasks:
            aiex_d.dma_free_task(task)

        return ctxt, mlirBlock
