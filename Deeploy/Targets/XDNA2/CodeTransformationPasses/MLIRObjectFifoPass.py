# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass that creates ObjectFifos and declares external kernels.

Given an :class:`MLIRExecutionBlock` with ``computeTile``, ``shimTile``,
``operatorRepresentation``, and (optionally) ``patternMemoryConstraint``,
this pass:

1. Derives ``tileSize`` and ``numTiles`` (from tiling solver or fallback).
2. Creates one ``aie_d.object_fifo`` per input tensor (shim → compute)
   and one per output tensor (compute → shim), all with depth 2
   (double-buffering).
3. Declares the external kernel via ``aie_d.external_func``.
4. Stores FIFO names, types, and kernel metadata on the block for
   downstream passes and the compute template.

The pass is operator-agnostic — it only needs the tensor names and a
tile-size derivation function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import aie.ir as ir
import numpy as np
from aie.dialects import aie as aie_d

from Deeploy.Logging import DEFAULT_LOGGER as log
from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext

MAX_TILE_SIZE = 1024


def _deriveTileShape(numElements: int, patternMemoryConstraint) -> Tuple[int, ...]:
    """Extract the N-D tile shape from the tiling solution.

    Returns the solver's ``l1Constraint.shape`` directly.
    """

    nodeConstraint = patternMemoryConstraint.nodeConstraints[0]
    outputConstraints = nodeConstraint.outputTensorMemoryConstraints
    if outputConstraints:
        firstOutputName = list(outputConstraints.keys())[0]
        tensorConstraint = outputConstraints[firstOutputName]
        if "L1" in tensorConstraint.memoryConstraints:
            l1Constraint = tensorConstraint.memoryConstraints["L1"]
            if l1Constraint.shape is not None:
                return tuple(int(d) for d in l1Constraint.shape)

    raise ValueError


class MLIRObjectFifoPass(MLIRCodeTransformationPass):
    """Create ObjectFifos and declare the external kernel.

    All operator-specific metadata (tensor keys, kernel function name,
    kernel object file, argument types) is read from the
    :class:`MLIRNodeTemplate` stored on ``mlirBlock.template``.

    Parameters
    ----------
    fifoDepth : int
        ObjectFifo depth (default 2 for double-buffering).
    """

    def __init__(self, fifoDepth: int = 2) -> None:
        self.fifoDepth = fifoDepth

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        template = mlirBlock.template
        inputTensorKeys = template.INPUT_KEYS
        outputTensorKeys = template.OUTPUT_KEYS

        opRepr = mlirBlock.operatorRepresentation
        numElements = int(opRepr['size'])

        # Read tile shape from the tiling solver.  The tile constraints
        # are the sole authority on valid shapes.
        tileShape = _deriveTileShape(numElements, mlirBlock.patternMemoryConstraint)
        tileSize = int(np.prod(tileShape))

        assert numElements % tileSize == 0, (f"[XDNA2] Tile size {tileSize} (shape {tileShape}) does not evenly "
                                             f"divide numElements {numElements}.  Fix the tile constraint.")

        numTiles = numElements // tileSize

        mlirBlock.tileShape = tileShape
        mlirBlock.tileSize = tileSize
        mlirBlock.numTiles = numTiles
        mlirBlock.numElements = numElements
        mlirBlock.kernelFuncName = template.KERNEL_FN
        mlirBlock.kernelObjFile = template.KERNEL_OBJ

        log.info(f"[XDNA2] ObjectFifo: tileShape={tileShape}, tileSize={tileSize}, "
                 f"numTiles={numTiles}, numElements={numElements}")

        # ObjectFifo memref is 1-D (flat) — the DMA and kernels work on flat buffers.
        tileTy = ir.MemRefType.get((tileSize,), ir.BF16Type.get())
        computeTile = mlirBlock.computeTile
        shimTile = mlirBlock.shimTile

        # Create input ObjectFifos (shim → compute)
        for idx, key in enumerate(inputTensorKeys):
            fifoName = f"in{idx + 1}_0"
            aie_d.object_fifo(fifoName, shimTile, [computeTile], self.fifoDepth, tileTy)
            mlirBlock.fifoMap[key] = fifoName
            mlirBlock.fifoTypes[key] = tileTy

        # Create output ObjectFifos (compute → shim)
        for idx, key in enumerate(outputTensorKeys):
            fifoName = f"out_{idx}"
            aie_d.object_fifo(fifoName, computeTile, [shimTile], self.fifoDepth, tileTy)
            mlirBlock.fifoMap[key] = fifoName
            mlirBlock.fifoTypes[key] = tileTy

        # Declare external kernel
        argTypes = template.kernelArgTypes(tileTy)
        aie_d.external_func(template.KERNEL_FN, argTypes, link_with = template.KERNEL_OBJ)

        return ctxt, mlirBlock
