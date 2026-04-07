# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Device-phase pass that emits the AIE core block with tiling loops.

This pass constructs the structural MLIR around the compute kernel:

1. Opens an ``@aie_d.core`` block linked to the kernel object file.
2. Opens an infinite outer ``scf.for`` loop (streaming).
3. Opens an inner ``scf.for`` tiling loop (``numTiles`` iterations).
4. Acquires input/output ObjectFifo elements.
5. Builds a modified ``operatorRepresentation`` where tensor keys
   (e.g. ``data_in_1``) are replaced with the acquired MLIR memref
   values and ``size`` is replaced with the tile size — mirroring
   how ``TilingVariableReplacement`` rewrites buffer names for C
   backends.
6. Calls ``template.emit(modifiedOpRepr)`` — the template only emits
   its ``func_d.call`` using values from ``operatorRepresentation``.
7. Releases all FIFO elements and closes loops.

The pass is operator-agnostic: it only needs the tensor key lists and
reads everything else from the :class:`MLIRExecutionBlock` populated by
prior passes (e.g. :class:`MLIRObjectFifoPass`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from aie.dialects import aie as aie_d
from aie.dialects import scf as scf_d

from Deeploy.MLIRDataTypes import MLIRCodeTransformationPass, MLIRExecutionBlock

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import NetworkContext


class MLIRComputeCorePass(MLIRCodeTransformationPass):
    """Emit ``@aie_d.core`` with tiling loops and FIFO acquire/release.

    The template stored on ``mlirBlock.template`` is called inside the
    inner loop with a *modified* ``operatorRepresentation`` whose tensor
    entries point to acquired MLIR memref values instead of buffer name
    strings.

    Parameters
    ----------
    inputTensorKeys : list of str
        Keys in ``operatorRepresentation`` that name input tensors.
    outputTensorKeys : list of str
        Keys that name output tensors.
    """

    def __init__(self, inputTensorKeys: List[str], outputTensorKeys: List[str]) -> None:
        self.inputTensorKeys = inputTensorKeys
        self.outputTensorKeys = outputTensorKeys

    def apply(self, ctxt: NetworkContext, mlirBlock: MLIRExecutionBlock,
              name: str) -> Tuple[NetworkContext, MLIRExecutionBlock]:
        computeTile = mlirBlock.computeTile
        kernelObj = mlirBlock.kernelObjFile
        tileSize = mlirBlock.tileSize
        numTiles = mlirBlock.numTiles
        opRepr = mlirBlock.operatorRepresentation
        template = mlirBlock.template

        # Use the first tensor's type as representative tile memref type
        firstKey = self.inputTensorKeys[0]
        tileTy = mlirBlock.fifoTypes[firstKey]

        @aie_d.core(computeTile)
        def _core():
            subviewTy = aie_d.ObjectFifoSubviewType.get(tileTy)
            for _ in scf_d.for_(0, 0x7FFFFFFFFFFFFFFF, 1):
                for _ in scf_d.for_(0, numTiles, 1):
                    # Acquire all input FIFO elements
                    acquiredElements = {}
                    for key in self.inputTensorKeys:
                        fifoName = mlirBlock.fifoMap[key]
                        acq = aie_d.objectfifo_acquire(subviewTy, aie_d.ObjectFifoPort.Consume, fifoName, 1)
                        acquiredElements[key] = aie_d.objectfifo_subview_access(tileTy, acq, 0)

                    # Acquire all output FIFO elements
                    for key in self.outputTensorKeys:
                        fifoName = mlirBlock.fifoMap[key]
                        acq = aie_d.objectfifo_acquire(subviewTy, aie_d.ObjectFifoPort.Produce, fifoName, 1)
                        acquiredElements[key] = aie_d.objectfifo_subview_access(tileTy, acq, 0)

                    # Build modified opRepr: replace tensor names with MLIR
                    # values, replace size with tile size.  This mirrors the
                    # C backend's TilingVariableReplacement pass.
                    modifiedOpRepr = {**opRepr, 'size': tileSize, **acquiredElements}

                    # Call the template — it only emits func_d.call()
                    template.emit(modifiedOpRepr)

                    # Release all inputs
                    for key in self.inputTensorKeys:
                        aie_d.objectfifo_release(aie_d.ObjectFifoPort.Consume, mlirBlock.fifoMap[key], 1)
                    # Release all outputs
                    for key in self.outputTensorKeys:
                        aie_d.objectfifo_release(aie_d.ObjectFifoPort.Produce, mlirBlock.fifoMap[key], 1)

                    scf_d.yield_([])
                scf_d.yield_([])

        return ctxt, mlirBlock
