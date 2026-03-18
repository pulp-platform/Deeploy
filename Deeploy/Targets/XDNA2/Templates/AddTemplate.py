# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 elementwise Add — compute kernel only.

This template emits only the AIE core compute logic (FIFO
acquire → kernel call → FIFO release).  ObjectFifo creation, external
kernel declaration, and DMA runtime-sequence configuration are handled
by :class:`MLIRObjectFifoPass` and :class:`MLIRRuntimeSequencePass`
respectively, which populate the :class:`MLIRExecutionBlock` before
this template's :meth:`emit` is called.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aie.dialects import aie as aie_d
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d
from aie.dialects import scf as scf_d
import aie.ir as ir

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2AddTemplate(MLIRNodeTemplate):
    """Compute-only MLIR template for BF16 elementwise Add on XDNA2 (AIE2p).

    Emits an ``@aie_d.core`` block containing nested loops that acquire
    input/output ObjectFifo elements and call the vectorised
    ``eltwise_add_bf16_vector`` kernel.

    All ObjectFifo creation and DMA configuration is performed by
    upstream :class:`MLIRCodeTransformationPass` instances.  This
    template reads FIFO names, tile size, and kernel metadata from the
    :class:`MLIRExecutionBlock` passed through ``kwargs['executionBlock']``.
    """

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # MLIR emission — compute kernel only
    # ------------------------------------------------------------------

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Emit the AIE core compute block for a BF16 Add node.

        Must be called inside an ``@aie_d.device(...)`` region **after**
        the device-phase code-transformation passes have populated the
        :class:`MLIRExecutionBlock`.

        Expected keyword arguments
        --------------------------
        executionBlock : MLIRExecutionBlock
            Carries ``computeTile``, ``fifoMap``, ``fifoTypes``,
            ``tileSize``, ``numTiles``, ``kernelFuncName``, and
            ``kernelObjFile`` — all set by ``MLIRObjectFifoPass``.
        """
        eb = kwargs['executionBlock']

        computeTile = eb.computeTile
        tileSize = eb.tileSize
        numTiles = eb.numTiles
        kernelFn = eb.kernelFuncName
        kernelObj = eb.kernelObjFile

        # MemRef / scalar types
        tileTy = eb.fifoTypes[list(eb.fifoTypes.keys())[0]]
        i32 = ir.IntegerType.get_signless(32)

        # FIFO names (populated by MLIRObjectFifoPass)
        in1Fifo = eb.fifoMap['data_in_1']
        in2Fifo = eb.fifoMap['data_in_2']
        outFifo = eb.fifoMap['data_out']

        @aie_d.core(computeTile, link_with=kernelObj)
        def _core():
            subviewTy = aie_d.ObjectFifoSubviewType.get(tileTy)
            for _ in scf_d.for_(0, 0x7FFFFFFFFFFFFFFF, 1):
                for _ in scf_d.for_(0, numTiles, 1):
                    acqIn1 = aie_d.objectfifo_acquire(subviewTy, aie_d.ObjectFifoPort.Consume, in1Fifo, 1)
                    elemIn1 = aie_d.objectfifo_subview_access(tileTy, acqIn1, 0)
                    acqIn2 = aie_d.objectfifo_acquire(subviewTy, aie_d.ObjectFifoPort.Consume, in2Fifo, 1)
                    elemIn2 = aie_d.objectfifo_subview_access(tileTy, acqIn2, 0)
                    acqOut = aie_d.objectfifo_acquire(subviewTy, aie_d.ObjectFifoPort.Produce, outFifo, 1)
                    elemOut = aie_d.objectfifo_subview_access(tileTy, acqOut, 0)
                    sizeVal = arith_d.constant(i32, tileSize)
                    func_d.call([], kernelFn, [elemIn1, elemIn2, elemOut, sizeVal])
                    aie_d.objectfifo_release(aie_d.ObjectFifoPort.Consume, in1Fifo, 1)
                    aie_d.objectfifo_release(aie_d.ObjectFifoPort.Consume, in2Fifo, 1)
                    aie_d.objectfifo_release(aie_d.ObjectFifoPort.Produce, outFifo, 1)
                    scf_d.yield_([])
                scf_d.yield_([])


referenceTemplate = XDNA2AddTemplate()
