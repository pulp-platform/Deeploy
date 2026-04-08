# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 elementwise Add — pure compute primitive.

This template emits **only** a ``func_d.call`` to the vectorised
``eltwise_add_bf16_vector`` kernel.  It receives its operands (acquired
ObjectFifo element memrefs) and tile size through
``operatorRepresentation``, exactly like a C Mako template receives
buffer-name strings.

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


class XDNA2AddTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 elementwise Add on XDNA2.

    ``emit()`` is called by :class:`MLIRComputeCorePass` inside an
    already-open ``@aie_d.core`` + tiling-loop context, with
    ``operatorRepresentation`` entries replaced by live MLIR values:

    * ``data_in_1``, ``data_in_2``, ``data_out`` — acquired memref
      elements (from ObjectFifo acquire).
    * ``size`` — tile size (Python int).
    """

    KERNEL_FN = "eltwise_add_bf16_vector"
    KERNEL_OBJ = "add.o"
    INPUT_KEYS = ['data_in_1', 'data_in_2']
    OUTPUT_KEYS = ['data_out']

    def __init__(self):
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Emit a single ``func.call`` to the vectorised Add kernel."""
        i32 = ir.IntegerType.get_signless(32)
        sizeVal = arith_d.constant(i32, int(operatorRepresentation['size']))
        func_d.call([], self.KERNEL_FN, [
            operatorRepresentation['data_in_1'],
            operatorRepresentation['data_in_2'],
            operatorRepresentation['data_out'],
            sizeVal,
        ])


referenceTemplate = XDNA2AddTemplate()
