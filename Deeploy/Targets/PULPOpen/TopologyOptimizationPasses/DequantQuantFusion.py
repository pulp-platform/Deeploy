# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _scalar(attr) -> float:
    return float(np.asarray(attr.values if isinstance(attr, gs.Constant) else attr).item())


def _constInput(node: gs.Node):
    return next((t for t in node.inputs if isinstance(t, gs.Constant)), None)


def _producer(tensor: gs.Tensor):
    return tensor.inputs[0] if len(tensor.inputs) == 1 else None


def _skipBiasAdd(node):
    if node is not None and node.op == 'Add' and _constInput(node) is not None:
        return _producer(next(t for t in node.inputs if not isinstance(t, gs.Constant)))
    return node


def _weightRows(layer: gs.Node) -> np.ndarray:
    w = layer.inputs[1].values
    if layer.op == 'MatMul' or (layer.op == 'Gemm' and not layer.attrs.get('transB', 0)):
        w = w.T
    return np.abs(w.reshape(w.shape[0], -1))


def _isLayer(node) -> bool:
    return node is not None and node.op in ('MatMul', 'Gemm', 'Conv') and isinstance(node.inputs[1], gs.Constant)


def _absMax(tensor: gs.Tensor) -> float:
    node = _producer(tensor)
    if node is None:
        return 2.0**31
    if node.op == 'Quant':
        return 2.0**_scalar(node.attrs['bit_width'])
    if node.op == 'RequantShift':
        return _scalar(node.attrs.get('n_levels_out', node.attrs.get('n_levels')))
    if node.op in ('Reshape', 'Transpose', 'Flatten', 'Squeeze', 'Unsqueeze', 'MaxPool', 'Pad'):
        return _absMax(node.inputs[0])
    if node.op == 'Add' and _constInput(node) is not None:
        x = next(t for t in node.inputs if not isinstance(t, gs.Constant))
        return _absMax(x) + float(np.abs(_constInput(node).values).max())
    if _isLayer(node):
        bias = node.inputs[2] if len(node.inputs) > 2 and isinstance(node.inputs[2], gs.Constant) else None
        return _absMax(node.inputs[0]) * float(_weightRows(node).sum(axis = 1).max()) + \
            (float(np.abs(bias.values).max()) if bias is not None else 0.0)
    return 2.0**24


def _channels(tensor: gs.Tensor) -> int:
    layer = _skipBiasAdd(_producer(tensor))
    if _isLayer(layer):
        return _weightRows(layer).shape[0]
    return int(tensor.shape[1]) if tensor.shape is not None and len(tensor.shape) > 1 else 1


def _fixedPoint(M: float, zIn: int, zOut: int, xMax: float, offset: np.ndarray):
    for log2D in range(31, 0, -1):
        mul = int(round(M * 2**log2D))
        add = zOut * 2**log2D - zIn * mul + np.round(offset * 2**log2D).astype(np.int64)
        if mul > 0 and xMax * mul + float(np.abs(add).max()) + 2**(log2D - 1) < 2**31:
            return mul, add, log2D
    return None


def _isSameGrid(tensor: gs.Tensor, bits: int, signed: int) -> bool:
    node = _producer(tensor)
    return node is not None and node.op == 'Quant' and \
        (_scalar(node.attrs['bit_width']), _scalar(node.attrs['signed'])) == (bits, signed)


def _replaceWithRequantShift(graph: gs.Graph, dq: gs.Node, q: gs.Node, name: str, bias = None):
    x = dq.inputs[0]

    zIn, zOut = _scalar(dq.attrs['zero_point']), _scalar(q.attrs['zero_point'])
    if zIn != round(zIn) or zOut != round(zOut):
        return graph

    # Deeploy's Quant node stores 1/scale
    M = _scalar(dq.attrs['scale']) * _scalar(q.attrs['scale'])
    bits, signed = int(_scalar(q.attrs['bit_width'])), int(_scalar(q.attrs['signed']))

    C = _channels(x)
    offset = np.zeros(C)
    if bias is not None:
        values = np.asarray(bias.values, dtype = np.float64).reshape(-1)
        if values.size not in (1, C):
            return graph
        offset = offset + values * _scalar(q.attrs['scale'])

    fp = _fixedPoint(M, int(zIn), int(zOut), _absMax(x), offset)
    if fp is None:
        return graph
    mul, addVec, log2D = fp

    if bias is None and mul == 2**log2D and not np.any(addVec) and _isSameGrid(x, bits, signed):
        graph.deleteNode(q)
        graph.deleteNode(dq)
        return graph

    conv = _producer(x)
    if conv is not None and conv.op == 'Conv' and len(conv.inputs) == 3 and \
            isinstance(conv.inputs[2], gs.Constant) and len(x.outputs) == 1:
        addVec = addVec + mul * np.round(conv.inputs[2].values.reshape(-1)).astype(np.int64)
        conv.inputs.pop(2)

    attrs = {
        'div': gs.Constant(name + '_div', np.array(2**log2D)),
        'n_levels_out': gs.Constant(name + '_n_levels', np.array(2**bits)),
        'signed': gs.Constant(name + '_signed', np.array(signed)),
    }
    _inputs = [
        x,
        gs.Constant(name + '_mul', np.full(C, mul, dtype = np.int32)),
        gs.Constant(name + '_add', addVec.astype(np.int32)),
    ]
    graph.replaceInsertNode(_inputs, q.outputs, gs.Node(op = 'RequantShift', name = name, attrs = attrs))
    return graph


def _dequant_quant_to_rqs_fun(graph: gs.Graph, match: Match, name: str):
    dq, q = [m for k, m in match.nodes_map.items()]
    if len(dq.outputs[0].outputs) != 1 or dq.outputs[0] in graph.outputs:
        return graph
    return _replaceWithRequantShift(graph, dq, q, name)


def _dequant_bias_quant_to_rqs_fun(graph: gs.Graph, match: Match, name: str):
    dq, add, q = [m for k, m in match.nodes_map.items()]
    bias = _constInput(add)
    if bias is None or len(dq.outputs[0].outputs) != 1 or len(add.outputs[0].outputs) != 1 or \
            dq.outputs[0] in graph.outputs or add.outputs[0] in graph.outputs:
        return graph
    return _replaceWithRequantShift(graph, dq, q, name, bias)


@contextagnostic
class DequantQuantToRequantShiftPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['dq_out'], op = 'Dequant', name = 'dequant')
        output = graph.layer(inputs = output, outputs = ['q_out'], op = 'Quant', name = 'quant')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_DEQUANT_QUANT_TO_RQS_PASS"
        super().__init__(graph, _dequant_quant_to_rqs_fun, name)


@contextagnostic
class DequantBiasQuantToRequantShiftPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['dq_out'], op = 'Dequant', name = 'dequant')
        output = graph.layer(inputs = output, outputs = ['add_out'], op = 'Add', name = 'bias_add')
        output = graph.layer(inputs = output, outputs = ['q_out'], op = 'Quant', name = 'quant')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_DEQUANT_BIAS_QUANT_TO_RQS_PASS"
        super().__init__(graph, _dequant_bias_quant_to_rqs_fun, name)
