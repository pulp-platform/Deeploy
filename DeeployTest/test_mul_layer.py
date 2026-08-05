# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import pytest

from Deeploy.Targets.Generic.Layers import MulLayer


@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,expected_shape",
    [
        ((2, 3), (), (2, 3)),
        ((2, 3), (1,), (2, 3)),
        ((2, 3), (2, 3), (2, 3)),
        ((2, 1, 4), (3, 4), (2, 3, 4)),
        ((5, 1, 7), (1, 3, 1), (5, 3, 7)),
        ((0, 3), (1, 3), (0, 3)),
    ],
)
def test_mul_compute_shapes_broadcasts_according_to_onnx(lhs_shape, rhs_shape, expected_shape):
    layer = MulLayer([])

    input_shapes, output_shapes = layer.computeShapes([lhs_shape, rhs_shape], [(2, 3)], {}, False)

    assert input_shapes == [expected_shape, expected_shape]
    assert output_shapes == [(2, 3)]


def test_mul_compute_shapes_rejects_incompatible_shapes():
    layer = MulLayer([])

    with pytest.raises(ValueError, match = "Cannot broadcast Mul input shapes"):
        layer.computeShapes([(2, 3), (4,)], [(2, 3)], {}, False)
