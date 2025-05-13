# ----------------------------------------------------------------------
#
# File: PULPPasses.py
#
# Last edited: 10.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
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

import copy
from collections import OrderedDict

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import BranchingMatcher, Match
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _squash_transpose_add_fun(graph: gs.Graph, match: Match, name: str):

    nodes_map = match.nodes_map

    # SCHEREMO: Check that perms are equal
    if not nodes_map['transpose1'].attrs['perm'] == nodes_map['transpose2'].attrs['perm']:
        return graph

    # SCHEREMO: Make sure we are requantizing layerwise
    if not (isinstance(nodes_map['add'].attrs['rqs1_add'], int)
            and isinstance(nodes_map['add'].attrs['rqs1_mul'], int)):
        return graph

    addNode = nodes_map['add']

    transposeAttrs = copy.deepcopy(nodes_map['transpose1'].attrs)
    newInputs = [nodes_map['transpose1'].inputs[0], nodes_map['transpose2'].inputs[0]]
    newOutputs = [addNode.outputs[0]]

    graph.deleteNode(nodes_map['transpose1'])
    graph.deleteNode(nodes_map['transpose2'])
    newAddOut = gs.Variable(name = addNode.outputs[0].name + "_tp")
    newAddOut.shape = newInputs[0].shape
    newAddOut.dtype = newOutputs[0].dtype

    addNode.outputs = [newAddOut]
    graph.layer(inputs = [newAddOut],
                outputs = [newOutputs[0]],
                op = "Transpose",
                name = addNode.name + "_transpose",
                attrs = transposeAttrs)

    #import IPython; IPython.embed()

    return graph


@contextagnostic
class RQAddTransposeSquashPass(ReplaceSequentialPatternPass):

    def __init__(self):
        _input1 = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')
        _addIn1 = gs.Variable(name = 'addIn1')
        _addIn2 = gs.Variable(name = 'addIn2')
        _addOut = gs.Variable(name = 'addOut')
        _rqs = gs.Variable(name = 'rqs')

        anyIn1 = gs.Node(inputs = [_input1], outputs = [_addIn1], op = r'Transpose', name = 'transpose1')
        anyIn2 = gs.Node(inputs = [_input2], outputs = [_addIn2], op = r'Transpose', name = 'transpose2')

        addOut = gs.Node(inputs = [_addIn1, _addIn2], outputs = [_addOut], op = 'RequantizedAdd', name = 'add')

        graph = gs.Graph(nodes = [anyIn1, anyIn2, addOut], inputs = [_input1, _input2], outputs = [_rqs])

        super().__init__(graph,
                         replacement_fn = _squash_transpose_add_fun,
                         name = "_SQUASH_TRANSPOSE_RQADD_PASS",
                         matcher = BranchingMatcher(regex_op = True))


def _merge_add_rq_fun(graph: gs.Graph, match: Match, name: str):

    nodes_map = match.nodes_map
    addNode = nodes_map['add']

    rqDict = OrderedDict([("rqs1", None), ("rqs2", None), ("rqsOut", None)])

    for key, node in nodes_map.items():

        if node.outputs[0].name == addNode.inputs[0].name:
            rqDict['rqs1'] = node
        elif node.outputs[0].name == addNode.inputs[1].name:
            rqDict['rqs2'] = node
        elif node.inputs[0].name == addNode.outputs[0].name:
            rqDict['rqsOut'] = node

    newAttrs = copy.copy(addNode.attrs)
    newInputs = []

    if rqDict['rqsOut'] is not None:
        newOutputs = rqDict['rqsOut'].outputs
    else:
        newOutputs = addNode.outputs

    defaultAttrs = {
        "mul": 1,
        "add": 0,
        "div": gs.Constant('div', np.array(1)),
        'shift': gs.Constant('div', np.array(0))
    }
    guessAttrs = {"n_levels_out": 256, "signed": np.array([True])}
    for idx, (rqKey, node) in enumerate(rqDict.items()):
        if node.op == "RequantShift":
            for key, attr in node.attrs.items():
                newAttrs[f"{rqKey}_{key}"] = attr

            if np.prod(node.inputs[1].values.shape) != 1:
                return graph

            if np.prod(node.inputs[2].values.shape) != 1:
                return graph

            if rqKey != 'rqsOut':
                newInputs.append(node.inputs[0])

            newAttrs[f"{rqKey}_mul"] = int(node.inputs[1].values.item())
            newAttrs[f"{rqKey}_add"] = int(node.inputs[2].values.item() + newAttrs[f"{rqKey}_div"].values.item() // 2)
            newAttrs[f"{rqKey}_shift"] = int(np.log2(newAttrs[f"{rqKey}_div"].values.item()))

        else:
            for key, attr in defaultAttrs.items():
                newAttrs[f"{rqKey}_{key}"] = attr

            for key, attr in guessAttrs.items():
                if not key in node.attrs:
                    newAttrs[f"{rqKey}_{key}"] = attr
                else:
                    newAttrs[f"{rqKey}_{key}"] = node.attrs[key]
            if rqKey != 'rqsOut':
                newInputs.append(addNode.inputs[idx])

    rqAdd = gs.Node(op = "RequantizedAdd", name = name, attrs = newAttrs)
    graph.replaceInsertNode(newInputs, newOutputs, rqAdd)

    return graph


@contextagnostic
class PULPAddRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        _input1 = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')
        _addIn1 = gs.Variable(name = 'addIn1')
        _addIn2 = gs.Variable(name = 'addIn2')
        _addOut = gs.Variable(name = 'addOut')
        _rqs = gs.Variable(name = 'rqs')

        anyIn1 = gs.Node(inputs = [_input1], outputs = [_addIn1], op = r'.*', name = 'any1')
        anyIn2 = gs.Node(inputs = [_input2], outputs = [_addIn2], op = r'.*', name = 'any2')

        addOut = gs.Node(inputs = [_addIn1, _addIn2], outputs = [_addOut], op = 'Add', name = 'add')
        output = gs.Node(inputs = [_addOut], outputs = [_rqs], op = r'RequantShift', name = 'rqsOut')

        graph = gs.Graph(nodes = [anyIn1, anyIn2, addOut, output], inputs = [_input1, _input2], outputs = [_rqs])

        super().__init__(graph,
                         replacement_fn = _merge_add_rq_fun,
                         name = "_MERGE_ADDRQ_PASS",
                         matcher = BranchingMatcher(regex_op = True))


def _merge_conv_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    conv = matched_nodes[0]
    rqs = matched_nodes[1]

    totalShift = int(np.log2(rqs.attrs['div'].values))

    # Artifically add half the shift division value to implement rounding
    rounding = 2**(totalShift - 1) if totalShift > 0 else 0

    rqs.inputs[-1].values = copy.deepcopy(rqs.inputs[-1].values) + rounding

    _inputs = list(conv.inputs) + list(rqs.inputs[1:])

    _outputs = rqs.outputs

    rqsConv = gs.Node(op = 'RequantizedConv', name = name, attrs = {**conv.attrs, **rqs.attrs, "shift": totalShift})
    graph.replaceInsertNode(_inputs, _outputs, rqsConv)

    return graph


@contextagnostic
class PULPConvRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['conv_out'], op = 'Conv', name = 'conv1')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_CONVRQ_PASS"
        super().__init__(graph, _merge_conv_rq_fun, name)


def _merge_gemm_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemm = matched_nodes[0]
    rqs = matched_nodes[1]

    totalShift = int(np.log2(rqs.attrs['div'].values))

    rqs.inputs[-1].values = copy.deepcopy(rqs.inputs[-1].values) + 2**(totalShift - 1)

    # GEMM has add
    if len(list(gemm.inputs)) == 3:

        gemm.inputs[2].values = np.round(gemm.inputs[2].values * (rqs.inputs[1].values)) + rqs.inputs[2].values

        #gemm.inputs[2].values = gemm.inputs[2].values + np.round(rqs.inputs[2].values / (rqs.inputs[1].values + 1e-3))
        # Keep input, weight from GEMM
        # Take mul from RQS
        _inputs = list(gemm.inputs) + list(rqs.inputs[1:2])
    else:
        _inputs = list(gemm.inputs) + list(rqs.inputs[2:]) + list(rqs.inputs[1:2])
    _outputs = rqs.outputs
    attrs = {**gemm.attrs, **rqs.attrs}
    attrs['shift'] = gs.Constant(name = 'shift', values = np.array(totalShift))
    #attrs['mul']=gs.Constant(name='mul',values = np.array(rqs.inputs[1].values))
    rqsGemm = gs.Node(op = 'RequantizedGemm', name = name, attrs = attrs)
    graph.replaceInsertNode(_inputs, _outputs, rqsGemm)

    return graph


@contextagnostic
class PULPGEMMRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['gemm_out'], op = 'Gemm', name = 'gemm')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_GEMMRQ_PASS"
        super().__init__(graph, _merge_gemm_rq_fun, name)


@contextagnostic
class PULPMatMulRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['gemm_out'], op = 'MatMul', name = 'gemm')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_GEMM_MATMUL_RQ_PASS"
        super().__init__(graph, _merge_gemm_rq_fun, name)
