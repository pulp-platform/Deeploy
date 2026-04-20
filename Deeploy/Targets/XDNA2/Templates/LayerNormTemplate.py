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

The kernel's ``cols`` argument is read from
``operatorRepresentation['lastDimLength']``, which the tile constraint
keeps constant (last dimension is never split).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    * ``size`` — tile size (flat element count, = ``lastDimLength``
      because the tile constraint forces single-row tiles).

    The tile constraint (:class:`XDNA2LayerNormTileConstraint`) forces
    all non-last dimensions to 1, so each tile is exactly one row.
    The kernel's ``cols`` argument therefore equals ``size``.
    """

    KERNEL_FN = "layer_norm_bf16_bf16_highacc"
    KERNEL_OBJ = "layernorm.o"
    INPUT_KEYS = ['data_in']
    OUTPUT_KEYS = ['data_out']

    def __init__(self):
        super().__init__()

    def emit(self, operatorRepresentation: OperatorRepresentation, **kwargs) -> None:
        """Emit a single ``func.call`` to the high-accuracy LayerNorm kernel.

        The kernel takes ``cols`` = number of elements per row.  Since
        the tile constraint guarantees single-row tiles, ``size``
        equals ``cols``.
        """
        i32 = ir.IntegerType.get_signless(32)
        colsVal = arith_d.constant(i32, int(operatorRepresentation['size']))
        func_d.call([], self.KERNEL_FN, [
            operatorRepresentation['data_in'],
            operatorRepresentation['data_out'],
            colsVal,
        ])


referenceTemplate = XDNA2LayerNormTemplate()
