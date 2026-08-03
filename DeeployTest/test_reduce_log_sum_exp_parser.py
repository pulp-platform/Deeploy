# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs
import pytest

from Deeploy.Targets.Generic.Parsers import ReduceLogSumExpParser


def _node(shape, axes_input = None, **attrs):
    inputs = [gs.Variable("data", dtype = np.float32, shape = shape)]
    if axes_input is not None:
        inputs.append(gs.Constant("axes", values = np.asarray(axes_input, dtype = np.int64)))
    output = gs.Variable("reduced", dtype = np.float32)
    return gs.Node(op = "ReduceLogSumExp", inputs = inputs, outputs = [output], attrs = attrs)


@pytest.mark.parametrize(
    "node,expected_axes,expected_shape",
    [
        (_node((2, 3, 4), axes_input = [-1]), [2], [2, 3, 1]),
        (_node((2, 3, 4), axes = [1], keepdims = 0), [1], [2, 4]),
        (_node((2, 3, 4)), [0, 1, 2], [1, 1, 1]),
        (_node((2, 3, 4), axes_input = [1, 2], keepdims = 0), [1, 2], [2]),
    ],
)
def test_reduce_log_sum_exp_parser_accepts_onnx_axes_forms(node, expected_axes, expected_shape):
    parser = ReduceLogSumExpParser()

    assert parser.parseNode(node)
    assert parser.operatorRepresentation["axes"].tolist() == expected_axes
    assert node.outputs[0].shape == expected_shape


@pytest.mark.parametrize(
    "node",
    [
        _node((2, 3, 4), axes_input = [0, 2]),
        _node((2, 3, 4), axes_input = [], noop_with_empty_axes = 1),
        _node((2, 3, 4), axes_input = [3]),
        _node((2, 3, 4), axes_input = [1, 1]),
    ],
)
def test_reduce_log_sum_exp_parser_rejects_unsupported_axes(node):
    assert not ReduceLogSumExpParser().parseNode(node)


def test_reduce_log_sum_exp_parser_flattens_consecutive_axes():
    reduction = ReduceLogSumExpParser._reductionShapeAndSizes((2, 3, 4, 5), [1, 2], keepdims = 0)

    assert reduction == ([2, 5], 2, 12, 5)
