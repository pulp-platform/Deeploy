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

from Deeploy.MLIRAIETypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext

MAX_TILE_SIZE = 1024


def _deriveTileSize(numElements: int, patternMemoryConstraint) -> int:
    """Extract tile size from the tiling solution, or fall back to MAX_TILE_SIZE."""
    tileSize = min(numElements, MAX_TILE_SIZE)

    if patternMemoryConstraint is not None:
        try:
            nodeConstraint = patternMemoryConstraint.nodeConstraints[0]
            outputConstraints = nodeConstraint.outputTensorMemoryConstraints
            if outputConstraints:
                firstOutputName = list(outputConstraints.keys())[0]
                tensorConstraint = outputConstraints[firstOutputName]
                if "L1" in tensorConstraint.memoryConstraints:
                    l1Constraint = tensorConstraint.memoryConstraints["L1"]
                    if l1Constraint.shape is not None:
                        tileSize = int(np.prod(l1Constraint.shape))
        except (AttributeError, IndexError, KeyError):
            pass

    # Ensure tile_size evenly divides num_elements
    if numElements % tileSize != 0:
        tileSize = max(d for d in range(1, tileSize + 1) if numElements % d == 0)

    return tileSize


class MLIRObjectFifoPass(MLIRCodeTransformationPass):
    """Create ObjectFifos and declare the external kernel.

    Parameters
    ----------
    inputTensorKeys : list of str
        Keys in ``operatorRepresentation`` that name input tensors
        (e.g. ``['data_in_1', 'data_in_2']``).
    outputTensorKeys : list of str
        Keys that name output tensors (e.g. ``['data_out']``).
    kernelFuncName : str
        Symbol name of the external AIE kernel function.
    kernelObjFile : str
        Object file to link with the AIE core (e.g. ``"add.o"``).
    kernelArgTypes : callable, optional
        A callable ``(tile_memref_type) -> list[ir.Type]`` that returns
        the kernel's argument types.  Defaults to
        ``[tile_ty, tile_ty, tile_ty, i32]`` (suitable for binary
        elementwise ops).
    fifoDepth : int
        ObjectFifo depth (default 2 for double-buffering).
    """

    def __init__(self,
                 inputTensorKeys: list,
                 outputTensorKeys: list,
                 kernelFuncName: str,
                 kernelObjFile: str,
                 kernelArgTypes = None,
                 fifoDepth: int = 2) -> None:
        self.inputTensorKeys = inputTensorKeys
        self.outputTensorKeys = outputTensorKeys
        self.kernelFuncName = kernelFuncName
        self.kernelObjFile = kernelObjFile
        self._kernelArgTypes = kernelArgTypes
        self.fifoDepth = fifoDepth

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        opRepr = mlirBlock.operatorRepresentation
        numElements = int(opRepr['size'])
        tileSize = _deriveTileSize(numElements, mlirBlock.patternMemoryConstraint)
        numTiles = numElements // tileSize

        mlirBlock.tileSize = tileSize
        mlirBlock.numTiles = numTiles
        mlirBlock.numElements = numElements
        mlirBlock.kernelFuncName = self.kernelFuncName
        mlirBlock.kernelObjFile = self.kernelObjFile

        tileTy = ir.MemRefType.get((tileSize,), ir.BF16Type.get())
        computeTile = mlirBlock.computeTile
        shimTile = mlirBlock.shimTile

        # Create input ObjectFifos (shim → compute)
        for idx, key in enumerate(self.inputTensorKeys):
            fifoName = f"in{idx + 1}_0"
            aie_d.object_fifo(fifoName, shimTile, [computeTile], self.fifoDepth, tileTy)
            mlirBlock.fifoMap[key] = fifoName
            mlirBlock.fifoTypes[key] = tileTy

        # Create output ObjectFifos (compute → shim)
        for idx, key in enumerate(self.outputTensorKeys):
            fifoName = f"out_{idx}"
            aie_d.object_fifo(fifoName, computeTile, [shimTile], self.fifoDepth, tileTy)
            mlirBlock.fifoMap[key] = fifoName
            mlirBlock.fifoTypes[key] = tileTy

        # Declare external kernel
        i32 = ir.IntegerType.get_signless(32)
        if self._kernelArgTypes is not None:
            argTypes = self._kernelArgTypes(tileTy)
        else:
            # Default: binary elementwise  (in1, in2, out, size)
            argTypes = [tileTy, tileTy, tileTy, i32]
        aie_d.external_func(self.kernelFuncName, argTypes, link_with = self.kernelObjFile)

        return ctxt, mlirBlock
