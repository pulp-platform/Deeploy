# ----------------------------------------------------------------------
#
# File: MemPoolPasses.py
#
# Last edited: 13.11.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
from typing import Dict, Union

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import BranchingMatcher, Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def merge_matmul_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    matmul = matched_nodes[0]
    rqs = matched_nodes[1]

    # WIESEP: Per element quantization is not supported for RQMatMul
    if len(rqs.inputs[2].shape) > 0 and rqs.inputs[2].shape[-1] != 1:
        return graph

    _inputs = list(matmul.inputs) + list(rqs.inputs[2:]) + list(rqs.inputs[1:2])
    _outputs = rqs.outputs

    attrs = {**matmul.attrs, **rqs.attrs}
    rqsMatMul = gs.Node(op = 'RQMatMul', name = name, attrs = attrs)
    graph.replaceInsertNode(_inputs, _outputs, rqsMatMul)

    return graph


@contextagnostic
class MemPoolMatMulRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['matmul_out'], op = 'MatMul', name = 'matmul')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = f"_MERGE_MATMUL_RQ_PASS"
        super().__init__(graph, merge_matmul_rq_fun, name)


def merge_gemm_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemm = matched_nodes[0]
    rqs = matched_nodes[1]

    # WIESEP: Per element quantization is not supported for RQGemm
    if len(rqs.inputs[2].shape) > 0 and rqs.inputs[2].shape[-1] != 1:
        return graph

    # WIESEP: Per column quantization is not supported for RQGemm
    if len(rqs.inputs[2].shape) > 2 and rqs.inputs[2].shape[-3] != 1:
        return graph

    _inputs = list(gemm.inputs) + list(rqs.inputs[2:]) + list(rqs.inputs[1:2])
    _outputs = rqs.outputs

    attrs = {**gemm.attrs, **rqs.attrs}
    rqsGemm = gs.Node(op = 'RQGemm', name = name, attrs = attrs)
    graph.replaceInsertNode(_inputs, _outputs, rqsGemm)

    return graph


@contextagnostic
class MemPoolGEMMRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['matmul_out'], op = 'Gemm', name = 'gemm')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = f"_MERGE_GEMM_RQ_PASS"
        super().__init__(graph, merge_gemm_rq_fun, name)


def _fuse_mhsa_fun(graph: gs.Graph, match: Match, name: str, batchedMatMul = False):
    # matched_nodes = [m for k, m in match.nodes_map.items()]

    def get_named_node(nodes_map: Dict, name: str) -> Union[gs.Node, None]:
        if name in nodes_map:
            return nodes_map[name]
        return None

    Projection_q = get_named_node(match.nodes_map, 'Projection_q')
    Bias_Pq = get_named_node(match.nodes_map, 'Bias_Pq')
    RequantShift_Pq = get_named_node(match.nodes_map, 'RequantShift_Pq')
    Reshape_Pq = get_named_node(match.nodes_map, 'Reshape_Pq')
    Transpose_Pq = get_named_node(match.nodes_map, 'Transpose_Pq')
    Projection_k = get_named_node(match.nodes_map, 'Projection_k')
    Bias_Pk = get_named_node(match.nodes_map, 'Bias_Pk')
    RequantShift_Pk = get_named_node(match.nodes_map, 'RequantShift_Pk')
    # Reshape_Pk = get_named_node(match.nodes_map, 'Reshape_Pk')
    # Transpose_Pk = get_named_node(match.nodes_map, 'Transpose_Pk')
    Projection_v = get_named_node(match.nodes_map, 'Projection_v')
    Bias_Pv = get_named_node(match.nodes_map, 'Bias_Pv')
    RequantShift_Pv = get_named_node(match.nodes_map, 'RequantShift_Pv')
    # Reshape_Pv = get_named_node(match.nodes_map, 'Reshape_Pv')
    Transpose_Pv = get_named_node(match.nodes_map, 'Transpose_Pv')
    # MatMul_a = get_named_node(match.nodes_map, 'MatMul_a')
    RequantShift_a = get_named_node(match.nodes_map, 'RequantShift_a')
    IntegerDiv_a = get_named_node(match.nodes_map, 'IntegerDiv_a')
    # Softmax_a = get_named_node(match.nodes_map, 'Softmax_a')
    # MatMul_o = get_named_node(match.nodes_map, 'MatMul_o')
    RequantShift_o = get_named_node(match.nodes_map, 'RequantShift_o')

    # Check if we accidentally swapped Q and K
    if Transpose_Pq.attrs['perm'] != Transpose_Pv.attrs['perm']:
        Projection_q = get_named_node(match.nodes_map, 'Projection_k')
        Bias_Pq = get_named_node(match.nodes_map, 'Bias_Pk')
        RequantShift_Pq = get_named_node(match.nodes_map, 'RequantShift_Pk')
        Reshape_Pq = get_named_node(match.nodes_map, 'Reshape_Pk')
        Transpose_Pq = get_named_node(match.nodes_map, 'Transpose_Pk')
        Projection_k = get_named_node(match.nodes_map, 'Projection_q')
        Bias_Pk = get_named_node(match.nodes_map, 'Bias_Pq')
        RequantShift_Pk = get_named_node(match.nodes_map, 'RequantShift_Pq')
        # Reshape_Pk = get_named_node(match.nodes_map, 'Reshape_Pq')
        # Transpose_Pk = get_named_node(match.nodes_map, 'Transpose_Pq')

        assert Transpose_Pq.attrs['perm'] == Transpose_Pv.attrs[
            'perm'], "[MemPoolFuseMHSAPass] MHSA key and value permutation is not the same!"

    attrs = {}
    H = Reshape_Pq.inputs[1].values[2]
    attrs['heads'] = H
    attrs['dim_head'] = Reshape_Pq.inputs[1].values[-1]  # Projection Size
    attrs['dim'] = Projection_q.inputs[0].shape[-2]  # Sequence Length
    attrs['S'] = Projection_q.inputs[0].shape[-2]  # Sequence Length
    attrs['E'] = Projection_q.inputs[1].shape[0]  # Embedding Size
    attrs['P'] = Reshape_Pq.inputs[1].values[-1]  # Projection Size

    attrs['wq_requant_mul'] = np.broadcast_to(RequantShift_Pq.inputs[1].values.reshape(-1), [H])
    attrs['wk_requant_mul'] = np.broadcast_to(RequantShift_Pk.inputs[1].values.reshape(-1), [H])
    attrs['wv_requant_mul'] = np.broadcast_to(RequantShift_Pv.inputs[1].values.reshape(-1), [H])

    # WIESEP: We also have to handle the integer div node!
    if IntegerDiv_a is not None:
        attrs['preattn_requant_mul'] = np.broadcast_to(
            np.round(RequantShift_a.inputs[1].values.reshape(-1) / IntegerDiv_a.inputs[1].values.reshape(-1)), [H])
    else:
        attrs['preattn_requant_mul'] = np.broadcast_to(RequantShift_a.inputs[1].values.reshape(-1), [H])

    attrs['postattn_requant_mul'] = np.broadcast_to(RequantShift_o.inputs[1].values.reshape(-1), [H])

    attrs['wq_requant_div'] = np.broadcast_to(RequantShift_Pq.attrs['div'].values.reshape(-1), [H])
    attrs['wk_requant_div'] = np.broadcast_to(RequantShift_Pk.attrs['div'].values.reshape(-1), [H])
    attrs['wv_requant_div'] = np.broadcast_to(RequantShift_Pv.attrs['div'].values.reshape(-1), [H])
    attrs['preattn_requant_div'] = np.broadcast_to(RequantShift_a.attrs['div'].values.reshape(-1), [H])
    attrs['postattn_requant_div'] = np.broadcast_to(RequantShift_o.attrs['div'].values.reshape(-1), [H])

    _inputs = []
    _inputs.append(Projection_q.inputs[0])
    _inputs.append(Projection_k.inputs[0])
    _inputs.append(Projection_v.inputs[0])

    def separate_heads(x: np.ndarray, heads: int, dim_head: int):
        return np.transpose(np.reshape(x, (-1, heads, dim_head)), (1, 0, 2))

    def get_constant_input(n: gs.Node):
        for input in n.inputs:
            if isinstance(input, gs.Constant):
                return input.values
        assert False, f"Did not find constant input for {n} node"

    def get_constant_input_or_zeros(n: gs.Node, shape):
        if n is None:
            return np.zeros(shape)
        else:
            return get_constant_input(n)

    # Transform from MUL-DIV-ADD to MUL-ADD-DIV
    attrs['wq_requant_add'] = np.broadcast_to(RequantShift_Pq.inputs[2].values.reshape(-1) // attrs['wq_requant_div'],
                                              [H])
    attrs['wk_requant_add'] = np.broadcast_to(RequantShift_Pk.inputs[2].values.reshape(-1) // attrs['wk_requant_div'],
                                              [H])
    attrs['wv_requant_add'] = np.broadcast_to(RequantShift_Pv.inputs[2].values.reshape(-1) // attrs['wv_requant_div'],
                                              [H])
    attrs['preattn_requant_add'] = np.broadcast_to(
        RequantShift_a.inputs[2].values.reshape(-1) // attrs['preattn_requant_div'], [H])
    attrs['postattn_requant_add'] = np.broadcast_to(
        RequantShift_o.inputs[2].values.reshape(-1) // attrs['postattn_requant_div'], [H])

    _inputs += [
        gs.Constant(name = name + '_wq_weight',
                    values = separate_heads(get_constant_input(Projection_q), attrs['heads'], attrs['P']))
    ]
    _inputs += [
        gs.Constant(name = name + '_wq_bias',
                    values = separate_heads(get_constant_input_or_zeros(Bias_Pq, (1, attrs['heads'], 1, attrs['P'])),
                                            attrs['heads'], attrs['P']))
    ]

    _inputs += [
        gs.Constant(name = name + '_wk_weight',
                    values = separate_heads(get_constant_input(Projection_k), attrs['heads'], attrs['P']))
    ]
    _inputs += [
        gs.Constant(name = name + '_wk_bias',
                    values = separate_heads(get_constant_input_or_zeros(Bias_Pk, (1, attrs['heads'], 1, attrs['P'])),
                                            attrs['heads'], attrs['P']))
    ]

    _inputs += [
        gs.Constant(name = name + '_wv_weight',
                    values = separate_heads(get_constant_input(Projection_v), attrs['heads'], attrs['P']))
    ]
    _inputs += [
        gs.Constant(name = name + '_wv_bias',
                    values = separate_heads(get_constant_input_or_zeros(Bias_Pv, (1, attrs['heads'], 1, attrs['P'])),
                                            attrs['heads'], attrs['P']))
    ]

    if batchedMatMul:
        Projection_Po = get_named_node(match.nodes_map, 'Projection_Po')
        Bias_Po = get_named_node(match.nodes_map, 'Bias_Po')
        RequantShift_Po = get_named_node(match.nodes_map, 'RequantShift_Po')

        attrs['n_levels'] = RequantShift_Po.attrs['n_levels_out'].values.reshape(1)
        attrs['signed'] = RequantShift_Po.attrs['signed'].values.reshape(1)
        attrs['wo_requant_mul'] = np.broadcast_to(RequantShift_Po.inputs[1].values.reshape(-1), [H])
        attrs['wo_requant_div'] = np.broadcast_to(RequantShift_Po.attrs['div'].values.reshape(-1), [H])
        attrs['wo_requant_add'] = np.broadcast_to(RequantShift_Po.inputs[2].values.reshape(-1),
                                                  [H]) // attrs['wo_requant_div']

        _inputs += [
            gs.Constant(name = name + '_wo_weight',
                        values = np.reshape(get_constant_input(Projection_Po), (attrs['heads'], attrs['P'], -1)))
        ]
        _inputs += [
            gs.Constant(name = name + '_wo_bias',
                        values = np.reshape(get_constant_input_or_zeros(Bias_Po, (1, attrs['heads'], 1, attrs['E'])),
                                            (attrs['heads'], attrs['E'])))
        ]

        _outputs = RequantShift_Po.outputs
        mhsa = gs.Node(op = 'MHSA', name = name, attrs = attrs)
        graph.replaceInsertNode(_inputs, _outputs, mhsa)
    else:
        # Extract ouptut projection for each head
        attrs['wo_requant_mul'] = np.empty((H))
        attrs['wo_requant_div'] = np.empty((H))
        attrs['wo_requant_add'] = np.empty((H))
        wo_weight = np.empty((H, attrs['P'], attrs['E']))
        wo_bias = np.empty((H, 1, attrs['E']))
        outputs = []
        for h in range(H):
            Gather_Po = get_named_node(match.nodes_map, f'Gather_o_{h}')
            index_h = int(get_constant_input(Gather_Po))
            MatMul_Po = get_named_node(match.nodes_map, f'MatMul_Po_{h}')
            Bias_Po = get_named_node(match.nodes_map, f'Bias_Po_{h}')
            RequantShift_Po = get_named_node(match.nodes_map, f'RequantShift_Po_{h}')
            outputs.append(RequantShift_Po.outputs[0])

            attrs['wo_requant_mul'][index_h] = RequantShift_Po.inputs[1].values.reshape(-1)
            attrs['wo_requant_div'][index_h] = RequantShift_Po.attrs['div'].values.reshape(-1)
            attrs['wo_requant_add'][index_h] = RequantShift_Po.inputs[2].values.reshape(
                -1) // attrs['wo_requant_div'][index_h]
            wo_weight[index_h] = get_constant_input(MatMul_Po)
            wo_bias[index_h] = get_constant_input_or_zeros(Bias_Po, (1, attrs['E']))

        _inputs += [gs.Constant(name = name + '_wo_weight', values = wo_weight)]
        _inputs += [gs.Constant(name = name + '_wo_bias', values = wo_bias)]

        # Extract final output
        attrs['n_levels'] = RequantShift_Po.attrs['n_levels_out'].values.reshape(1)
        attrs['signed'] = RequantShift_Po.attrs['signed'].values.reshape(1)

        if H > 1:
            _output = get_named_node(match.nodes_map, f'Add_Po_{H-2}').outputs[0]
        else:
            _output = get_named_node(match.nodes_map, f'RequantShift_Po_0').outputs[0]

        mhsa_out = graph.layer(inputs = _inputs, outputs = [name + f'_out'], op = 'MHSA', name = name, attrs = attrs)
        graph.layer(inputs = mhsa_out,
                    outputs = [_output],
                    op = 'ReduceSum',
                    name = name + "_sum",
                    attrs = {
                        'axes': [1],
                        "keepdims": "0"
                    })

        mhsa_out[0].shape = [_output.shape[0]] + [int(H)] + _output.shape[1:]
        mhsa_out[0].dtype = np.float32
        # Disconnect input nodes of all output tensors
        _output.inputs = _output.inputs[-1:]

        graph.cleanup().toposort()

    return graph


@contextagnostic
class MemPoolFuseMHSAPass(ReplaceSequentialPatternPass):

    def __init__(self, H, integerDiv = False, preSoftMaxRQ = True, bias = False):

        graph = gs.Graph()
        _input_q = gs.Variable(name = 'input_q')
        _input_k = gs.Variable(name = 'input_k')
        _input_v = gs.Variable(name = 'input_v')

        # Query Projection
        output_q = graph.layer(inputs = [_input_q], outputs = ['pQ'], op = 'MatMul', name = 'Projection_q')
        if bias:
            output_q = graph.layer(inputs = output_q, outputs = ['pQ_b'], op = 'Add', name = 'Bias_Pq')
        output_q = graph.layer(inputs = output_q, outputs = ['pQ_rq'], op = 'RequantShift', name = 'RequantShift_Pq')
        output_q = graph.layer(inputs = output_q, outputs = ['pQ_r'], op = 'Reshape', name = 'Reshape_Pq')
        output_q = graph.layer(inputs = output_q, outputs = ['pQ_t'], op = 'Transpose', name = 'Transpose_Pq')

        # Key Projection
        output_k = graph.layer(inputs = [_input_k], outputs = ['pK'], op = 'MatMul', name = 'Projection_k')
        if bias:
            output_k = graph.layer(inputs = output_k, outputs = ['pK_b'], op = 'Add', name = 'Bias_Pk')
        output_k = graph.layer(inputs = output_k, outputs = ['pK_rq'], op = 'RequantShift', name = 'RequantShift_Pk')
        output_k = graph.layer(inputs = output_k, outputs = ['pK_r'], op = 'Reshape', name = 'Reshape_Pk')
        output_k = graph.layer(inputs = output_k, outputs = ['pK_t'], op = 'Transpose', name = 'Transpose_Pk')

        # Value Projection
        output_v = graph.layer(inputs = [_input_v], outputs = ['pV'], op = 'MatMul', name = 'Projection_v')
        if bias:
            output_v = graph.layer(inputs = output_v, outputs = ['pV_b'], op = 'Add', name = 'Bias_Pv')
        output_v = graph.layer(inputs = output_v, outputs = ['pV_rq'], op = 'RequantShift', name = 'RequantShift_Pv')
        output_v = graph.layer(inputs = output_v, outputs = ['pV_r'], op = 'Reshape', name = 'Reshape_Pv')
        output_v = graph.layer(inputs = output_v, outputs = ['pV_t'], op = 'Transpose', name = 'Transpose_Pv')

        # Attention Matrix
        output_a = graph.layer(inputs = output_q + output_k, outputs = ['a'], op = 'MatMul', name = 'MatMul_a')
        if preSoftMaxRQ:
            output_a = graph.layer(inputs = output_a, outputs = ['a_rq'], op = 'RequantShift', name = 'RequantShift_a')

        if integerDiv:
            output_a = graph.layer(inputs = output_a, outputs = ['a_d'], op = 'IntegerDiv', name = 'IntegerDiv_a')

        output_a = graph.layer(inputs = output_a, outputs = ['a_s'], op = 'ITAPartialMax', name = 'Softmax_a')

        # Attention
        output = graph.layer(inputs = output_v + output_a, outputs = ['o'], op = 'MatMul', name = 'MatMul_o')

        if H == -1:
            # WIESEP: This only works if the output projection is a batched matrix multiplication
            output = graph.layer(inputs = output, outputs = ['o_rq'], op = 'RequantShift', name = 'RequantShift_o')
            # output = graph.layer(inputs = output, outputs = ['o_t'], op = 'Transpose', name = 'Transpose_o')
            # output = graph.layer(inputs = output, outputs = ['o_r'], op = 'Reshape', name = 'Reshape_Po')
            output = graph.layer(inputs = output, outputs = ['pO'], op = 'MatMul', name = 'Projection_Po')
            if bias:
                output = graph.layer(inputs = output, outputs = ['pO_b'], op = 'Add', name = 'Bias_Po')
            output = graph.layer(inputs = output, outputs = ['pO_rq'], op = 'RequantShift', name = 'RequantShift_Po')
        else:
            attention = graph.layer(inputs = output, outputs = ['o_rq'], op = 'RequantShift', name = 'RequantShift_o')

            projection_out = []
            for i in range(H):
                output = graph.layer(inputs = attention, outputs = [f'o_{i}'], op = 'Gather', name = f'Gather_o_{i}')
                output = graph.layer(inputs = output, outputs = [f'pO_{i}'], op = 'MatMul', name = f'MatMul_Po_{i}')
                if bias:
                    output = graph.layer(inputs = output, outputs = ['pO_{i}_b'], op = 'Add', name = f'Bias_Po_{i}')
                output = graph.layer(inputs = output,
                                     outputs = [f'pO_{i}_rq'],
                                     op = 'RequantShift',
                                     name = f'RequantShift_Po_{i}')
                projection_out.extend(output)

            for i in range(H - 1):
                if i == 0:
                    inp = [projection_out[0], projection_out[i + 1]]
                else:
                    inp = [output[0], projection_out[i + 1]]

                output = graph.layer(inputs = inp, outputs = [f'pO_sum_{i}'], op = 'Add', name = f'Add_Po_{i}')

        graph.outputs.append(output[0])
        graph.inputs.append(_input_q)
        graph.inputs.append(_input_k)
        graph.inputs.append(_input_v)

        name = "_FUSE_MHSA_PASS"

        # WIESEP: Debug Export pattern graph to ONNX
        # model = gs.export_onnx(graph, False)
        # onnx.save(model, f'pattern_{name}.onnx')

        super().__init__(graph, partial(_fuse_mhsa_fun, batchedMatMul = (H == -1)), name, matcher = BranchingMatcher())


def _split_mhsa_fun(graph: gs.Graph, match: Match, name: str):

    def get_named_node(nodes_map: Dict, name: str) -> gs.Node:
        if name in nodes_map:
            return nodes_map[name]
        raise KeyError(f"Did not find node with name {name}")

    MHSA = get_named_node(match.nodes_map, 'MHSA')

    input_q = MHSA.inputs[0]
    input_kv = MHSA.inputs[1]
    ReduceSum = get_named_node(match.nodes_map, 'ReduceSum')
    RequantShift = get_named_node(match.nodes_map, 'RequantShift')

    _outputs = RequantShift.outputs

    # Calculate the number of heads
    H = int(MHSA.attrs['heads'])
    S = int(MHSA.attrs['dim'])
    E = input_kv.shape[-1]
    P = int(MHSA.attrs['dim_head'])

    if H == 1:
        return graph

    # Create a list to hold the output nodes
    mhsa_outputs = []

    def extractHead(H: int, i: int):
        _attrs = {}
        _attrs['dim'] = MHSA.attrs['dim']
        _attrs['dim_head'] = MHSA.attrs['dim_head']
        _attrs['heads'] = H
        _attrs['n_levels'] = MHSA.attrs['n_levels']
        _attrs['signed'] = MHSA.attrs['signed']

        _attr_names = [
            "wq_requant_add", "wk_requant_add", "wv_requant_add", "wo_requant_add", "preattn_requant_add",
            "postattn_requant_add", "wq_requant_mul", "wk_requant_mul", "wv_requant_mul", "wo_requant_mul",
            "preattn_requant_mul", "postattn_requant_mul", "wq_requant_div", "wk_requant_div", "wv_requant_div",
            "wo_requant_div", "preattn_requant_div", "postattn_requant_div"
        ]
        for att in _attr_names:
            _attrs[att] = MHSA.attrs[att][i:i + H]

        _inputs = [input_q, input_kv, input_kv]
        _inputs_names = ["wq_weight", "wq_bias", "wk_weight", "wk_bias", "wv_weight", "wv_bias", "wo_weight", "wo_bias"]
        for idx, inp in enumerate(_inputs_names):
            _inputs += [
                gs.Constant(name = name + f'_MHSA_H{i}_{i+H-1}_{inp}', values = MHSA.inputs[idx + 3].values[i:i + H])
            ]

        # Create a new MHSA node for the current set of 4 heads
        mhsa_out = graph.layer(inputs = _inputs,
                               outputs = [name + f'_MHSA_H{i}_{i+H-1}_out'],
                               op = 'MHSA',
                               name = name + f'_MHSA_H{i}_{i+H-1}',
                               attrs = _attrs)

        # Append the new MHSA node to the output nodes list
        output_sum = graph.layer(inputs = mhsa_out,
                                 outputs = [name + f'_MHSA_H{i}_{i+H-1}_out_sum'],
                                 op = 'ReduceSum',
                                 name = name + f'_ReduceSum_H{i}_{i+H-1}',
                                 attrs = ReduceSum.attrs)
        mhsa_outputs.extend(output_sum)

    # Split the MHSA node into multiple MHSA nodes calculating 4, 2 or 1 heads each
    for i in range(0, H, 4):
        if i + 4 <= H:
            extractHead(4, i)
        elif i + 3 <= H:
            extractHead(2, i)
            extractHead(1, i + 2)
        elif i + 2 <= H:
            extractHead(2, i)
        else:
            extractHead(1, i)

    # Add layer to sum all heads
    out_sum = graph.layer(inputs = mhsa_outputs, outputs = [name + '_Out_sum'], op = 'Add', name = name + '_Add')

    # Add final requantiztion step
    RequantShift.inputs = out_sum + RequantShift.inputs[1:]
    # graph.layer(inputs=out_sum, outputs = _outputs, op = 'RequantShift', name = name + '_RequantShift')

    graph.deleteNode(MHSA)
    # graph.deleteNode(ReduceSum)
    # graph.deleteNode(RequantShift)

    graph.cleanup().toposort()

    return graph


@contextagnostic
class MemPoolSplitMHSAPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input_q = gs.Variable(name = 'input_q')
        _input_kv = gs.Variable(name = 'input_kv')

        output = graph.layer(inputs = [_input_q, _input_kv], outputs = ['Out'], op = 'MHSA', name = 'MHSA')

        output = graph.layer(inputs = output, outputs = ['Out_sum'], op = 'ReduceSum', name = 'ReduceSum')
        output = graph.layer(inputs = output, outputs = ['Out_sum_rqs'], op = 'RequantShift', name = 'RequantShift')

        graph.outputs.append(output)
        graph.inputs.append(_input_q)
        graph.inputs.append(_input_kv)

        name = "_SPLIT_MHSA_PASS"
        super().__init__(graph, _split_mhsa_fun, name, matcher = NonBranchingMatcher())
