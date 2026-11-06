# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from Deeploy.DeeployTypes import ConstantBuffer, NetworkContext, VariableBuffer
from Deeploy.Targets.Generic.Layers import MulLayer
from Deeploy.Targets.Generic.Parsers import MulParser

# Mark all tests in this module as deeploy_internal
pytestmark = pytest.mark.deeploy_internal


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
def test_mul_broadcast_shape_follows_onnx(lhs_shape, rhs_shape, expected_shape):
    assert MulLayer.broadcastShape(lhs_shape, rhs_shape) == expected_shape
    assert MulLayer.broadcastShape(rhs_shape, lhs_shape) == expected_shape


# Without constness information (computeShapes called on its own), every operand is treated as non-constant: it is
# only relabeled when that does not change its element count.
@pytest.mark.parametrize(
    "lhs_shape,rhs_shape,expected_lhs,expected_rhs",
    [
        ((2, 3), (), (2, 3), (1,)),
        ((2, 3), (1,), (2, 3), (1,)),
        ((), (2, 3), (1,), (2, 3)),
        ((1, 1), (2, 3), (1,), (2, 3)),
        ((2, 3), (2, 3), (2, 3), (2, 3)),
        ((2, 3), (1, 2, 3), (1, 2, 3), (1, 2, 3)),
        ((2, 1, 4), (3, 4), (2, 1, 4), (3, 4)),
        ((2, 3, 4), (3, 4), (2, 3, 4), (3, 4)),
        ((5, 1, 7), (1, 3, 1), (5, 1, 7), (1, 3, 1)),
    ],
)
def test_mul_compute_shapes_keeps_real_shape_of_variables(lhs_shape, rhs_shape, expected_lhs, expected_rhs):
    layer = MulLayer([])

    input_shapes, output_shapes = layer.computeShapes([lhs_shape, rhs_shape], [(7,)], {}, False)

    assert input_shapes == [expected_lhs, expected_rhs]
    assert output_shapes == [(7,)]


def test_mul_compute_shapes_rejects_incompatible_shapes():
    layer = MulLayer([])

    with pytest.raises(ValueError, match = "Cannot broadcast Mul input shapes"):
        layer.computeShapes([(2, 3), (4,)], [(2, 3)], {}, False)


def _mulContext(shape_a, shape_b, const_a = False, const_b = False):
    """Graph and context for C = A * B, as the deployer builds them before broadcasting."""
    rng = np.random.default_rng(0)
    ctxt = NetworkContext(VariableBuffer, ConstantBuffer, None, None)
    values = {"A": rng.standard_normal(shape_a), "B": rng.standard_normal(shape_b)}
    tensors = {}
    for name, shape, is_const in (("A", shape_a, const_a), ("B", shape_b, const_b)):
        if is_const:
            tensors[name] = gs.Constant(name, values[name])
            ctxt.add(ConstantBuffer(name, list(shape), values[name]), "global")
        else:
            tensors[name] = gs.Variable(name, np.float32, shape)
            ctxt.add(VariableBuffer(name, list(shape)), "local")
    out_shape = np.broadcast_shapes(shape_a, shape_b)
    tensors["C"] = gs.Variable("C", np.float32, out_shape)
    ctxt.add(VariableBuffer("C", list(out_shape)), "local")
    node = gs.Node("Mul", "mul", inputs = [tensors["A"], tensors["B"]], outputs = [tensors["C"]])
    return ctxt, node, values


def _broadcastAndParse(ctxt, node):
    parser = MulParser()
    layer = MulLayer([])(node)
    layer.mapper = type("_Mapper", (), {"parser": parser})()
    assert parser.parseNode(node)
    ctxt = layer.broadcast(ctxt)
    ctxt, ok = parser.parseNodeCtxt(ctxt, node)
    return ctxt, ok, parser.operatorRepresentation


def _realSize(ctxt, name):
    buffer = ctxt.lookup(name)
    if isinstance(buffer, ConstantBuffer):
        return int(np.asarray(buffer.values).size)
    return int(np.prod(buffer.shape))


@pytest.mark.parametrize(
    "shape_a,shape_b,const_a,const_b,expect_swap,expected_size_b",
    [
        # element-wise
        ((2, 3, 4), (2, 3, 4), False, False, False, 24),
        ((2, 3, 4), (3, 4), False, True, False, 24),  # constant B is materialized to the output shape
        ((2, 3, 4), (1, 2, 3, 4), False, False, False, 24),  # same element count, only relabeled
        # scalar B
        ((2, 3, 4), (1,), False, False, False, 1),
        ((2, 3, 4), (), False, True, False, 1),
        # scalar A: swapped into B
        ((1,), (2, 3, 4), False, False, True, 1),
        ((), (2, 3, 4), True, False, True, 1),
        # both scalar
        ((1,), (1, 1), False, False, False, 1),
    ],
)
def test_mul_parser_accepts_supported_patterns(shape_a, shape_b, const_a, const_b, expect_swap, expected_size_b):
    ctxt, node, values = _mulContext(shape_a, shape_b, const_a, const_b)

    ctxt, ok, rep = _broadcastAndParse(ctxt, node)

    assert ok
    assert (rep['A'], rep['B']) == (("B", "A") if expect_swap else ("A", "B"))
    assert rep['C'] == "C"
    assert rep['size'] == int(np.prod(np.broadcast_shapes(shape_a, shape_b)))
    assert rep['sizeB'] == expected_size_b

    # Every index the kernels produce (i < size, and 0 or i for B) is inside the stored data.
    assert rep['size'] <= _realSize(ctxt, rep['A'])
    assert (1 if rep['sizeB'] == 1 else rep['size']) <= _realSize(ctxt, rep['B'])

    # Element-wise kernel semantics on the stored data match ONNX broadcasting.
    lookup = {
        name: np.asarray(ctxt.lookup(name).values) if const else values[name]
        for name, const in (("A", const_a), ("B", const_b))
    }
    a = lookup[rep['A']].reshape(-1)
    b = lookup[rep['B']].reshape(-1)
    c = np.array([a[i] * b[0 if rep['sizeB'] == 1 else i] for i in range(rep['size'])])
    np.testing.assert_allclose(c, np.multiply(values["A"], values["B"]).reshape(-1))


@pytest.mark.parametrize(
    "shape_a,shape_b,const_a,const_b",
    [
        ((2, 1, 4), (3, 4), False, False),  # variable broadcast on both sides
        ((2, 1, 4), (3, 4), False, True),  # constant B is materialized, but variable A is still too small
        ((2, 3, 4), (3, 4), False, False),  # variable suffix broadcast
        ((3, 4), (2, 3, 4), False, False),
        ((2, 3, 1), (2, 3, 4), False, False),
        ((0, 3), (1, 3), False, False),  # zero-size output
    ],
)
def test_mul_parser_rejects_unsupported_broadcast(shape_a, shape_b, const_a, const_b):
    ctxt, node, _ = _mulContext(shape_a, shape_b, const_a, const_b)

    ctxt, ok, _ = _broadcastAndParse(ctxt, node)

    assert not ok
    # Non-constant operands keep the shape of the data they actually hold.
    for name, shape, is_const in (("A", shape_a, const_a), ("B", shape_b, const_b)):
        if not is_const and np.prod(shape) != np.prod(np.broadcast_shapes(shape_a, shape_b)):
            assert int(np.prod(ctxt.lookup(name).shape)) == max(int(np.prod(shape)), 1)
