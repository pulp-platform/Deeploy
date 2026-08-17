# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Tuple

import numpy as np

from Deeploy.DeeployTypes import Shape
from Deeploy.Targets.Generic.Layers import AddLayer, MulLayer


class _ScalarPreservingShapeMixin:
    """Keep a single-element second operand at its original shape.

    The Generic Add and Mul layers rewrite the shorter operand's shape to the
    longer one, which expresses broadcasting notionally but does not
    materialise any data. For a genuine scalar that is harmful on Snitch: the
    kernels broadcast input2[0] themselves, selected by the is_scalar flag the
    parser derives from the operand shape. Once the shape has been rewritten
    the flag comes out false, the buffer is allocated for the full tensor, and
    the kernel reads elements that were never written.
    """

    def computeShapes(self, inputShapes: Shape, outputShapes: Shape, operatorRepresentation,
                      channels_first) -> Tuple[Shape, Shape]:

        if len(inputShapes) > 1 and np.prod(inputShapes[1]) == 1:
            return (inputShapes, [inputShapes[0]])

        return super().computeShapes(inputShapes, outputShapes, operatorRepresentation, channels_first)


class SnitchAddLayer(_ScalarPreservingShapeMixin, AddLayer):
    pass


class SnitchMulLayer(_ScalarPreservingShapeMixin, MulLayer):
    pass
