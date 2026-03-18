# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 elementwise Add.

Uses ``aie.dialects`` (from the pip-installed ``mlir-aie`` package) to emit
verified MLIR operations into an existing module context provided by the
:class:`XDNA2Deployer`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aie.dialects import aie as aie_d
from aie.dialects import aiex as aiex_d
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d
from aie.dialects import scf as scf_d
import aie.ir as ir

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2AddTemplate(MLIRNodeTemplate):
    """MLIR template for BF16 elementwise Add on XDNA2 (AIE2p).

    The :meth:`emit` method constructs a single-core AIE program with:

    * Two input ObjectFifos and one output ObjectFifo (depth 2 for
      double-buffering).
    * A compute core that loops, acquiring / releasing FIFO elements and
      calling the vectorised ``eltwise_add_bf16_vector`` kernel.
    * A runtime sequence that configures shim DMA for L3 ↔ L1 transfers.

    Parameters are extracted from the *operatorRepresentation* populated
    by the parser (``size`` = total number of BF16 elements).
    """

    KERNEL_FN = "eltwise_add_bf16_vector"
    KERNEL_OBJ = "add.o"
    MAX_TILE_SIZE = 1024

    def __init__(self):
        super().__init__()

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def getAIEParams(self, operatorRepresentation: OperatorRepresentation,
                     tilingConstraint=None) -> dict:
        """Extract AIE parameters from the operator representation.
        
        If tilingConstraint is available (tiling enabled), use information
        from it. Otherwise fall back to fixed tile sizes.

        Parameters
        ----------
        operatorRepresentation : OperatorRepresentation
            Parsed operator representation containing 'size' (total elements).
        tilingConstraint : PatternMemoryConstraints, optional
            Tiling solution from the solver. If provided, tile size is derived
            from the tiling solution.

        Returns
        -------
        dict
            ``num_elements``, ``tile_size`` (from tiling solution if available,
            otherwise clamped to MAX_TILE_SIZE).
        """
        num_elements = int(operatorRepresentation['size'])
        
        # If tiling is enabled, extract tile size from the tiling solution
        if tilingConstraint is not None:
            # tilingConstraint is a PatternMemoryConstraints with nodeConstraints
            nodeConstraint = tilingConstraint.nodeConstraints[0]
            outputConstraints = nodeConstraint.outputTensorMemoryConstraints
            if outputConstraints:
                # Get the first output tensor's L1 memory constraint (tile shape)
                firstOutputName = list(outputConstraints.keys())[0]
                tensorConstraint = outputConstraints[firstOutputName]
                # Use L1 constraint which holds the tile shape for the AIE core
                if "L1" in tensorConstraint.memoryConstraints:
                    l1Constraint = tensorConstraint.memoryConstraints["L1"]
                    if l1Constraint.shape is not None:
                        tile_size = int(np.prod(l1Constraint.shape))
                    else:
                        tile_size = min(num_elements, self.MAX_TILE_SIZE)
                else:
                    tile_size = min(num_elements, self.MAX_TILE_SIZE)
            else:
                tile_size = min(num_elements, self.MAX_TILE_SIZE)
        else:
            tile_size = min(num_elements, self.MAX_TILE_SIZE)
            
        if num_elements % tile_size != 0:
            # Round down to the largest divisor of num_elements that fits
            tile_size = max(d for d in range(1, tile_size + 1) if num_elements % d == 0)
            
        return {
            'num_elements': num_elements,
            'tile_size': tile_size,
        }

    # ------------------------------------------------------------------
    # MLIR emission
    # ------------------------------------------------------------------

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Add AIE operations for a BF16 Add node into the current device context.

        Must be called inside an ``@aie_d.device(...)`` region (the deployer
        sets this up).  The following keyword arguments are expected:

        * ``compute_tile`` — result of ``aie_d.tile(col, row)``
        * ``shim_tile`` — result of ``aie_d.tile(col, 0)``
        * ``tilingConstraint`` — optional NodeMemoryConstraint for tiled execution

        Parameters
        ----------
        operatorRepresentation : OperatorRepresentation
            Parsed operator representation with 'size' and other attributes
        **kwargs
            compute_tile, shim_tile, tilingConstraint (optional)
        """
        tilingConstraint = kwargs.get('tilingConstraint', None)
        params = self.getAIEParams(operatorRepresentation, tilingConstraint=tilingConstraint)
        num_elements = params['num_elements']
        tile_size = params['tile_size']
        num_tiles = num_elements // tile_size

        compute_tile = kwargs['compute_tile']
        shim_tile = kwargs['shim_tile']

        # MemRef types
        tile_ty = ir.MemRefType.get((tile_size,), ir.BF16Type.get())
        i32 = ir.IntegerType.get_signless(32)

        # ObjectFifos (depth 2 for double-buffering)
        aie_d.object_fifo("in1_0", shim_tile, [compute_tile], 2, tile_ty)
        aie_d.object_fifo("in2_0", shim_tile, [compute_tile], 2, tile_ty)
        aie_d.object_fifo("out_0", compute_tile, [shim_tile], 2, tile_ty)

        # External kernel declaration
        aie_d.external_func(self.KERNEL_FN, [tile_ty, tile_ty, tile_ty, i32])

        # Compute core
        @aie_d.core(compute_tile, link_with=self.KERNEL_OBJ)
        def _core():
            subview_ty = aie_d.ObjectFifoSubviewType.get(tile_ty)
            for _ in scf_d.for_(0, 0x7FFFFFFFFFFFFFFF, 1):
                for _ in scf_d.for_(0, num_tiles, 1):
                    acq_in1 = aie_d.objectfifo_acquire(subview_ty, aie_d.ObjectFifoPort.Consume, "in1_0", 1)
                    elem_in1 = aie_d.objectfifo_subview_access(tile_ty, acq_in1, 0)
                    acq_in2 = aie_d.objectfifo_acquire(subview_ty, aie_d.ObjectFifoPort.Consume, "in2_0", 1)
                    elem_in2 = aie_d.objectfifo_subview_access(tile_ty, acq_in2, 0)
                    acq_out = aie_d.objectfifo_acquire(subview_ty, aie_d.ObjectFifoPort.Produce, "out_0", 1)
                    elem_out = aie_d.objectfifo_subview_access(tile_ty, acq_out, 0)
                    size_val = arith_d.constant(i32, tile_size)
                    func_d.call([], self.KERNEL_FN, [elem_in1, elem_in2, elem_out, size_val])
                    aie_d.objectfifo_release(aie_d.ObjectFifoPort.Consume, "in1_0", 1)
                    aie_d.objectfifo_release(aie_d.ObjectFifoPort.Consume, "in2_0", 1)
                    aie_d.objectfifo_release(aie_d.ObjectFifoPort.Produce, "out_0", 1)
                    scf_d.yield_([])
                scf_d.yield_([])

    def emitRuntimeSequence(self, operatorRepresentation: OperatorRepresentation,
                            seq_args: list, tilingConstraint=None) -> None:
        """Emit DMA configuration inside a runtime_sequence block.

        Parameters
        ----------
        operatorRepresentation : OperatorRepresentation
            Node representation (used to extract ``num_elements``).
        seq_args : list
            Block arguments of the runtime_sequence (memref values for
            in1, in2, out — in the order matching the ONNX graph I/O).
        tilingConstraint : NodeMemoryConstraint, optional
            Tiling solution from the solver (currently ignored, for future use).
        """
        params = self.getAIEParams(operatorRepresentation, tilingConstraint=tilingConstraint)
        num_elements = params['num_elements']

        dims = [
            aie_d.bd_dim_layout(size=1, stride=0),
            aie_d.bd_dim_layout(size=1, stride=0),
            aie_d.bd_dim_layout(size=1, stride=0),
            aie_d.bd_dim_layout(size=num_elements, stride=1),
        ]

        in1, in2, out = seq_args[0], seq_args[1], seq_args[2]

        task_in1 = aiex_d.dma_configure_task_for("in1_0")
        block_in1 = task_in1.body.blocks.append()
        with ir.InsertionPoint(block_in1):
            aie_d.dma_bd(in1, offset=0, len=num_elements, dimensions=dims, burst_length=0)
            aie_d.end()
        aiex_d.dma_start_task(task_in1)

        task_in2 = aiex_d.dma_configure_task_for("in2_0")
        block_in2 = task_in2.body.blocks.append()
        with ir.InsertionPoint(block_in2):
            aie_d.dma_bd(in2, offset=0, len=num_elements, dimensions=dims, burst_length=0)
            aie_d.end()
        aiex_d.dma_start_task(task_in2)

        task_out = aiex_d.dma_configure_task_for("out_0", issue_token=True)
        block_out = task_out.body.blocks.append()
        with ir.InsertionPoint(block_out):
            aie_d.dma_bd(out, offset=0, len=num_elements, dimensions=dims, burst_length=0)
            aie_d.end()
        aiex_d.dma_start_task(task_out)
        aiex_d.dma_await_task(task_out)
        aiex_d.dma_free_task(task_in1)
        aiex_d.dma_free_task(task_in2)


referenceTemplate = XDNA2AddTemplate()
