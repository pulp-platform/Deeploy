# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""XDNA2 MLIR template for BF16 LayerNorm (high-accuracy) — pure compute primitive.

This template emits **only** a ``func_d.call`` to the vectorised
``layer_norm_bf16_bf16_highacc`` kernel.  The kernel normalises one
row at a time (``lastDimLength`` elements), with gamma=1 and beta=0
hardcoded.

Weight and bias tensors parsed by :class:`LayerNormParser` are **not**
streamed via ObjectFifos — only ``data_in`` and ``data_out`` appear in
:attr:`INPUT_KEYS` / :attr:`OUTPUT_KEYS`.

The kernel's ``cols`` argument equals the FIFO tile size, which is
forced to ``lastDimLength`` via :meth:`deriveTileSize`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

import aie.ir as ir
from aie.dialects import arith as arith_d
from aie.dialects import func as func_d

from Deeploy.MLIRDataTypes import MLIRNodeTemplate

if TYPE_CHECKING:
    from Deeploy.DeeployTypes import OperatorRepresentation


class XDNA2LayerNormTemplate(MLIRNodeTemplate):
    """Pure compute-primitive for BF16 LayerNorm (high-accuracy) on XDNA2.

    ``emit()`` is called by :class:`MLIRComputeCorePass` inside an
    already-open ``@aie_d.core`` + tiling-loop context, with
    ``operatorRepresentation`` entries replaced by live MLIR values:

    * ``data_in`` — acquired memref element (from ObjectFifo acquire).
    * ``data_out`` — acquired memref element (from ObjectFifo acquire).
    * ``size`` — tile size (= ``lastDimLength``).
    """

    KERNEL_FN = "layer_norm_bf16_bf16_highacc"
    KERNEL_OBJ = "layernorm.o"
    INPUT_KEYS = ['data_in']
    OUTPUT_KEYS = ['data_out']

    def __init__(self):
        super().__init__()

    def deriveTileSize(self, numElements: int, patternMemoryConstraint: Any,
                       operatorRepresentation: 'OperatorRepresentation') -> int:
        """Force tile size = lastDimLength so each tile is one row.

        The LayerNorm kernel processes exactly one row of ``cols``
        elements.  Tiling must not split or merge rows.
        """
        return int(operatorRepresentation['lastDimLength'])

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Emit a single ``func.call`` to the high-accuracy LayerNorm kernel."""
        i32 = ir.IntegerType.get_signless(32)
        sizeVal = arith_d.constant(i32, int(operatorRepresentation['size']))
        func_d.call([], self.KERNEL_FN, [
            operatorRepresentation['data_in'],
            operatorRepresentation['data_out'],
            sizeVal,
        ])


referenceTemplate = XDNA2LayerNormTemplate()
