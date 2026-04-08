# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 SiLU (x * sigmoid(x)) — pure compute primitive.

This template emits **only** a ``func_d.call`` to the vectorised
``silu_bf16`` kernel.  It receives its operands (acquired ObjectFifo
element memrefs) and tile size through ``operatorRepresentation``,
exactly like the Add template.

All structural MLIR (``@aie_d.core``, loops, FIFO acquire/release,
ObjectFifo creation, DMA configuration) is handled by
:class:`MLIRCodeTransformationPass` instances upstream.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aie.ir as ir
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2SiLUTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 SiLU on XDNA2.

    ``emit()`` is called by :class:`MLIRComputeCorePass` inside an
    already-open ``@aie_d.core`` + tiling-loop context, with
    ``operatorRepresentation`` entries replaced by live MLIR values:

    * ``data_in`` — acquired memref element (from ObjectFifo acquire).
    * ``data_out`` — acquired memref element (from ObjectFifo acquire).
    * ``size`` — tile size (Python int).
    """

    KERNEL_FN = "silu_bf16"

    def __init__(self):
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Emit a single ``func.call`` to the vectorised SiLU kernel."""
        i32 = ir.IntegerType.get_signless(32)
        sizeVal = arith_d.constant(i32, int(operatorRepresentation['size']))
        func_d.call([], self.KERNEL_FN, [
            operatorRepresentation['data_in'],
            operatorRepresentation['data_out'],
            sizeVal,
        ])


referenceTemplate = XDNA2SiLUTemplate()
