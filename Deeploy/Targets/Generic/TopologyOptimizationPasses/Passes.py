# ----------------------------------------------------------------------
#
# File: BasicPasses.py
#
# Last edited: 28.04.2023
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
from functools import partial
from typing import List

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import BranchingMatcher, Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _merge_trueintegerdiv_rq_fun(graph: gs.Graph, match: Match, name: str):

    matched_nodes = [m for k, m in match.nodes_map.items()]

    integerDiv = matched_nodes[0]
    rqs2 = matched_nodes[1]

    rqs2Add = rqs2.inputs[2]
    rqs2Mul = rqs2.inputs[1]
    rqs2Div = rqs2.attrs["div"]

    if not isinstance(rqs2Add, gs.Constant) or np.prod(rqs2Add.shape) > 1:
        return graph

    if not rqs2Add.values.item() == 0:
        return graph

    if not isinstance(rqs2Mul, gs.Constant) or np.prod(rqs2Mul.shape) > 1:
        return graph

    Delta = integerDiv.attrs['Delta']
    eta = integerDiv.attrs['eta']
    eps = integerDiv.attrs['eps']
    y = integerDiv.attrs['y']

    stretch = 2**8

    coeff = np.floor(((Delta * eta) / (y * eta + eps))) * stretch

    rqs2Mul.values = np.round(rqs2Mul.values * coeff)
    rqs2.attrs['div'].values = rqs2.attrs['div'].values * stretch

    _inputs = [*integerDiv.inputs[:1], *rqs2.inputs[1:]]
    _outputs = rqs2.outputs

    newRQS = gs.Node(op = "RequantShift", name = rqs2.name + "_repl", attrs = {**rqs2.attrs})

    graph.replaceInsertNode(_inputs, _outputs, newRQS)

    return graph


@contextagnostic
class MergeTrueIntegerDivRequantShiftPass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input],
                             outputs = ['integerdiv_out'],
                             op = 'TrueIntegerDiv',
                             name = 'integerdiv')
        output = graph.layer(inputs = output, outputs = ['rqs_2'], op = 'RequantShift', name = 'rqs2')

        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_TRUEINTEGERDIV_PASS"
        super().__init__(graph, _merge_trueintegerdiv_rq_fun, name)


def _merge_integerdiv_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    integerdiv = matched_nodes[0]
    rqs = matched_nodes[1]
    totalShift = np.round(np.log2(rqs.attrs['div'].values))

    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values + 1e-3)  # normalize add

    shiftNode = gs.Constant(f'{integerdiv.name}_shift', np.array(totalShift))
    _inputs = list(integerdiv.inputs) + list(rqs.inputs[1:]) + [shiftNode]
    _outputs = rqs.outputs

    rqsIntegerDiv = gs.Node(op = 'RQIntegerDiv', name = name, attrs = {**integerdiv.attrs, **rqs.attrs})
    graph.replaceInsertNode(_inputs, _outputs, rqsIntegerDiv)

    return graph


@contextagnostic
class IntegerDivRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['integerdiv_out'], op = 'IntegerDiv', name = 'integerdiv')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_INTEGERDIV_PASS"
        super().__init__(graph, _merge_integerdiv_rq_fun, name)


def _merge_ihardswish_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    ihardswish = matched_nodes[0]
    rqs = matched_nodes[1]
    totalShift = np.round(np.log2(rqs.attrs['div'].values))

    if not len(rqs.inputs) == 3:
        return

    if not (rqs.inputs[1].shape == [] or np.prod(rqs.inputs[1].shape) == 1):
        return

    if not (rqs.inputs[2].shape == [] or np.prod(rqs.inputs[2].shape) == 1):
        return

    if not isinstance(rqs.inputs[1], gs.Constant):
        return

    if not isinstance(rqs.inputs[2], gs.Constant):
        return

    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values + 1e-3)  # normalize add

    requantArgs = {"mul": rqs.inputs[1].values.item(), "add": rqs.inputs[2].values.item(), "shift": totalShift}

    _inputs = list(ihardswish.inputs)
    _outputs = rqs.outputs

    rqsiHardswish = gs.Node(op = 'RequantizediHardswish',
                            name = name,
                            attrs = {
                                **ihardswish.attrs,
                                **rqs.attrs,
                                **requantArgs
                            })
    graph.replaceInsertNode(_inputs, _outputs, rqsiHardswish)

    return graph


@contextagnostic
class iHardswishRequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['ihardswish_out'], op = 'iHardswish', name = 'ihardswish')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = f"_MERGE_iHARDSWISHRQ_PASS"
        super().__init__(graph, _merge_ihardswish_rq_fun, name)


def _merge_igelu_rq_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    igelu = matched_nodes[0]
    rqs = matched_nodes[1]
    totalShift = np.round(np.log2(rqs.attrs['div'].values))

    rqs.inputs[-1].values = np.round(rqs.inputs[-1].values / rqs.inputs[-2].values + 1e-3)  # normalize add

    shiftNode = gs.Constant(f'{igelu.name}_shift', np.array(totalShift))
    _inputs = list(igelu.inputs) + list(rqs.inputs[1:]) + [shiftNode]
    _outputs = rqs.outputs

    #import IPython; IPython.embed()

    rqsiGELU = gs.Node(op = 'RequantizediGELU', name = name, attrs = {**igelu.attrs, **rqs.attrs})
    graph.replaceInsertNode(_inputs, _outputs, rqsiGELU)

    return graph


@contextagnostic
class iGELURequantMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['igelu_out'], op = 'iGELU', name = 'igelu')
        output = graph.layer(inputs = output, outputs = ['rqs'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = f"_MERGE_iGELURQ_PASS"
        super().__init__(graph, _merge_igelu_rq_fun, name)


def _merge_rqs_add_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    add = matched_nodes[0]
    rqs = matched_nodes[1]

    if (isinstance(add.inputs[0], gs.Constant) or isinstance(add.inputs[1], gs.Constant)) and isinstance(
            rqs.inputs[2], gs.Constant):
        if isinstance(add.inputs[0], gs.Constant):
            idx = 1  # Non-constant idx
            constantTensor = add.inputs[0]
        else:
            idx = 0  # non-constant idx
            constantTensor = add.inputs[1]
        if constantTensor.values.shape != tuple(add.outputs[0].shape):
            rqs.inputs[2].values = (rqs.inputs[1].values * constantTensor.values) + rqs.inputs[2].values
            add.inputs[(idx + 1) % 2].values = add.inputs[(idx + 1) % 2].values * 0
            rqs.inputs[0] = add.inputs[idx]
        return graph
    else:
        return graph


@contextagnostic
class MergeConstAddAndRequantPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['add_out'], op = 'Add', name = 'add1')
        output = graph.layer(inputs = output, outputs = ['rqs_out'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_MERGE_RQS_ADD_PASS"
        super().__init__(graph, _merge_rqs_add_fun, name)


def _skip_rqs_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    node = matched_nodes[0]
    rqs = matched_nodes[1]

    # Check if it is a unity requant
    mul = rqs.inputs[1].values
    add = rqs.inputs[2].values
    if (rqs.attrs['div'].values == mul).all() and (add == 0).all():
        # Remove the requant node
        graph.replaceInsertNode(node.inputs, rqs.outputs, node)

    return graph


@contextagnostic
class SkipUnityRequantPass(ReplaceSequentialPatternPass):

    def __init__(self, previous_op_regex: str, num_inputs: int = 1):
        if previous_op_regex == "":
            raise ValueError('Operator not set!')

        graph = gs.Graph()
        inputs = [gs.Variable(name = f'input_{i}') for i in range(num_inputs)]
        output = graph.layer(inputs = inputs, outputs = ['op_out'], op = previous_op_regex)
        output = graph.layer(inputs = output, outputs = ['rqs_out'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs = inputs

        name = "_SKIP_RQS_PASS"
        super().__init__(graph, _skip_rqs_fun, name, NonBranchingMatcher(regex_op = True))


def _skip_emptyconcat_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    concat = matched_nodes[0]

    remove = False
    empty_inputs = []
    for inp in concat.inputs:
        # Check if one of the shapes is zero
        if np.prod(inp.shape) == 0:
            empty_inputs.append(inp)
            remove = True
            break

    if remove:
        # Check if one of the inputs is empty
        for inp in concat.inputs:
            # Check if one of the shapes is non-zero
            if np.prod(inp.shape) != 0:
                for outputNode in list(concat.outputs[0].outputs):
                    # Swap the outputTensor with inputTensor in the downstream nodes
                    outputNode.inputs[outputNode.inputs.index(concat.outputs[0])] = inp
                concat.inputs.clear()
                concat.outputs.clear()

                # Check if inputs are global inputs and remove them
                graph.inputs = [inp for inp in graph.inputs if inp not in empty_inputs]

                graph.cleanup().toposort()
                return graph

    return graph


@contextagnostic
class SkipEmptyConcatPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')
        output = graph.layer(inputs = [_input, _input2], outputs = ['concat_out'], op = 'Concat')
        graph.outputs.append(output)
        graph.inputs = [_input, _input2]

        name = "_SKIP_EMPTY_CONCAT_PASS"
        super().__init__(graph, _skip_emptyconcat_fun, name)


def _split_add_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    add = matched_nodes[0]

    inputs = add.inputs
    if len(inputs) > 2:
        result = [inputs[0]]
        for i in range(0, len(inputs) - 1):
            result = graph.layer(op = "Add",
                                 name = name + f'_Add{i}',
                                 inputs = [result[0], inputs[i + 1]],
                                 outputs = [name + f'_Add{i}_out'])

        add.outputs[0].outputs[0].inputs[0] = result[0]

        add.inputs.clear()
        add.outputs.clear()
        graph.cleanup().toposort()

    return graph


@contextagnostic
class SplitAddPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['add_out'], op = 'Add', name = 'add1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_SPLIT_ADD_PASS"
        super().__init__(graph, _split_add_fun, name)


def _extract_padding_fun(graph: gs.Graph, match: Match, name: str, value = 0):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    conv = matched_nodes[0]
    if 'pads' in conv.attrs and np.sum(conv.attrs['pads']) > 1:
        pads = copy.deepcopy(conv.attrs['pads'])
        shape = copy.deepcopy(conv.inputs[0].shape)
        newPads = np.zeros(2 * len(shape))
        assert len(shape) - 2 == len(pads) / 2, "Conv padding dims do not match!"
        newShape = shape

        beginPads = pads[0:len(pads) // 2]
        endPads = pads[len(pads) // 2:]
        for idx, i in enumerate(beginPads):
            newShape[2 + idx] = newShape[2 + idx] + i
            newPads[2 + idx] = i

        for idx, i in enumerate(endPads):
            newShape[2 + idx] = newShape[2 + idx] + i
            newPads[len(newPads) // 2 + 2 + idx] = i

        newConvInput = gs.Variable(name + '_padded_input', dtype = np.float32, shape = newShape)
        #valConst = gs.Constant('value', np.array(0))
        conv.attrs['pads'] = [0 for pad in conv.attrs['pads']]
        newPad = gs.Node(op = 'Pad',
                         name = name + '_pad',
                         attrs = {
                             'pads': newPads,
                             'mode': 'constant',
                             'value': value
                         },
                         inputs = [conv.inputs[0]],
                         outputs = [newConvInput])

        conv.inputs[0] = newConvInput
        graph.nodes.append(newPad)
        graph.cleanup().toposort()
        #import IPython; IPython.embed()

    return graph


@contextagnostic
class ExtractPaddingFromPoolPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['pool_out'], op = 'MaxPool', name = 'maxpool1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_EXTRACT_POOL_PASS"
        # SCHEREMO: This is a workaround!!!
        super().__init__(graph, partial(_extract_padding_fun, value = -128), name)


@contextagnostic
class ExtractPaddingFromConvPass(ReplaceSequentialPatternPass):

    def __init__(self):
        passes = []
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['conv_out'], op = 'Conv', name = 'conv1')
        graph.outputs.append(output)
        graph.inputs = [_input]

        name = "_EXTRACT_CONV_PASS"
        super().__init__(graph, _extract_padding_fun, name)


def _merge_matmul_add_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    gemm = matched_nodes[0]
    add = matched_nodes[1]
    _bias = add.inputs[0] if isinstance(add.inputs[0], gs.Constant) else add.inputs[1]
    _inputs = gemm.inputs + [_bias]
    _outputs = add.outputs

    rqsGemm = gs.Node(op = 'Gemm', name = name, attrs = {'alpha': 1.0, 'beta': 1.0})
    graph.replaceInsertNode(_inputs, _outputs, rqsGemm)

    return graph


@contextagnostic
class MatMulAddMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['gemm_out'], op = 'MatMul', name = 'gemm')
        output = graph.layer(inputs = output, outputs = ['add_out'], op = 'Add', name = 'add')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_MATMUL_ADD_PASS"
        super().__init__(graph, _merge_matmul_add_fun, name)


def _propagate_requant_fun(graph: gs.Graph, match: Match, name: str):

    matched_nodes = [m for k, m in match.nodes_map.items()]
    add = matched_nodes[0]
    rqs = matched_nodes[1]

    inputNode1 = add.inputs[0]
    inputNode2 = add.inputs[1]

    newAdd1 = gs.Constant(name = name + '_rqs1_add', values = rqs.inputs[2].values)
    newAdd2 = gs.Constant(name = name + '_rqs2_add', values = rqs.inputs[2].values)
    newMul1 = gs.Constant(name = name + '_rqs1_mul', values = rqs.inputs[1].values)
    newMul2 = gs.Constant(name = name + '_rqs2_mul', values = rqs.inputs[1].values)

    newAddInput1 = gs.Variable(name + '_add_in_1', dtype = np.float32, shape = inputNode1.shape)
    newAddInput2 = gs.Variable(name + '_add_in_2', dtype = np.float32, shape = inputNode2.shape)

    newRQS1 = gs.Node(op = 'RequantShift',
                      name = name + '_rqs1',
                      attrs = rqs.attrs,
                      inputs = [inputNode1, newMul1, newAdd1],
                      outputs = [newAddInput1])
    newRQS2 = gs.Node(op = 'RequantShift',
                      name = name + '_rqs2',
                      attrs = rqs.attrs,
                      inputs = [inputNode2, newMul2, newAdd2],
                      outputs = [newAddInput2])

    graph.nodes.append(newRQS1)
    graph.nodes.append(newRQS2)

    add.inputs = [newAddInput1, newAddInput2]
    graph.deleteNode(rqs)

    return graph


@contextagnostic
class PropagateRequantThroughAddPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')
        output = graph.layer(inputs = [_input, _input2], outputs = ['add_out'], op = 'Add', name = 'add1')
        output = graph.layer(inputs = output, outputs = ['r1_out'], op = 'RequantShift', name = 'r1')
        graph.outputs.append(output)
        graph.inputs = [_input, _input2]

        name = "_OPT_ADD_RQS_PASS"
        super().__init__(graph, _propagate_requant_fun, name)


def _merge_requant_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    attrs = {}
    rqs1 = matched_nodes[0]
    rqs2 = matched_nodes[1]

    div1 = rqs1.attrs['div'].values
    div2 = rqs2.attrs['div'].values
    newDiv = max(div1, div2)
    minDiv = min(div1, div2)
    nLevels = max(rqs1.attrs['n_levels_out'].values, rqs2.attrs['n_levels_out'].values)
    signed = max(rqs1.attrs['signed'].values, rqs2.attrs['signed'].values)

    attrs['div'] = gs.Constant(name = 'div', values = newDiv)
    attrs['n_levels'] = gs.Constant(name = 'n_levels', values = nLevels)
    attrs['signed'] = gs.Constant(name = 'signed', values = signed)

    if isinstance(rqs1.inputs[1], gs.Constant) and isinstance(rqs1.inputs[2], gs.Constant) and \
       isinstance(rqs2.inputs[1], gs.Constant) and isinstance(rqs2.inputs[2], gs.Constant):
        mul1 = rqs1.inputs[1].values
        mul2 = rqs2.inputs[1].values
        add1 = rqs1.inputs[2].values
        add2 = rqs2.inputs[2].values

        newMul = (mul1 * mul2)
        newAdd = (add1 * mul2) + (div1 * add2)

        newMul = gs.Constant(name = rqs1.name + name + '_mul', values = np.array(np.round(newMul / minDiv)))
        newAdd = gs.Constant(name = rqs1.name + name + '_add', values = np.array(np.round(newAdd / minDiv)))

        _inputs = [rqs1.inputs[0], newMul, newAdd]
        _outputs = rqs2.outputs
        newTrans = gs.Node(op = 'RequantShift', name = name, attrs = attrs)
        graph.replaceInsertNode(_inputs, _outputs, newTrans)
        return graph
    else:
        return graph


@contextagnostic
class MergeRequantPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['r1_out'], op = 'RequantShift', name = 'r1')
        output = graph.layer(inputs = output, outputs = ['r2_out'], op = 'RequantShift', name = 'r2')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_OPT_RQS_PASS"
        super().__init__(graph, _merge_requant_fun, name)


def _merge_transposes_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]
    t2 = matched_nodes[1]

    #Transpose forth and back - delete both nodes

    if (t1.inputs[0].shape == t2.outputs[0].shape):
        # Find Nodes-to-be-replaced
        graph.deleteNode(t2)
        graph.deleteNode(t1)
        graph.cleanup().toposort()
        return graph
    # Net the transpose
    else:
        p1 = t1.attrs['perm']
        p2 = t2.attrs['perm']
        newPerm = [p1[idx] for idx in p2]

    _inputs = list(t1.inputs)
    _outputs = list(t2.outputs)

    # Check if one of the intermedate nodes is a output node
    for node in t1.outputs:
        if node in graph.outputs:
            return graph

    newTrans = gs.Node(op = 'Transpose', name = name, attrs = {"perm": newPerm})
    graph.replaceInsertNode(_inputs, _outputs, newTrans)

    graph.cleanup().toposort()
    return graph


@contextagnostic
class TransposeMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'Transpose', name = 't1')
        output = graph.layer(inputs = output, outputs = ['t2_out'], op = 'Transpose', name = 't2')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_TRANSPOSES_PASS"
        super().__init__(graph, _merge_transposes_fun, name)


def _split_transposes_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]

    if len(t1.outputs[0].outputs) <= 1:
        return graph

    perm = t1.attrs['perm']
    inputVar = t1.inputs[0]
    inputNode = t1.inputs[0].inputs[0]

    originalNode = t1.outputs[0]

    postSplitOutput = gs.Variable(name = f"{t1.outputs[0].name}_split", dtype = np.float32, shape = t1.inputs[0].shape)
    inputNode.outputs = [postSplitOutput]

    for node in originalNode.outputs.copy():
        nodeName = node.name + f"_transpose_in"
        varName = node.name + f"_transpose_in_var"
        newOutput = gs.Variable(name = varName, dtype = np.float32, shape = t1.outputs[0].shape)

        transposeNode = gs.Node(name = nodeName,
                                op = "Transpose",
                                inputs = [postSplitOutput],
                                outputs = [newOutput],
                                attrs = {'perm': perm})

        graph.nodes.append(transposeNode)

        newNodeInputs = []
        for _input in node.inputs:
            if _input != originalNode:
                newNodeInputs.append(_input)
            else:
                newNodeInputs.append(newOutput)

        node.inputs = newNodeInputs

    t1.outputs = []
    t1.inputs = []

    graph.cleanup().toposort()
    return graph


@contextagnostic
class TransposeSplitPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'Transpose', name = 't1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_SPLIT_TRANSPOSES_PASS"
        super().__init__(graph, _split_transposes_fun, name)


def _const_perm_opt_transposes_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]

    perm = t1.attrs['perm']
    if all([idx == val for idx, val in enumerate(perm)]):
        graph.deleteNode(t1)

    return graph


@contextagnostic
class TransposeNoPermOptPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'Transpose', name = 't1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_CONST_PERM_OPT_TRANSPOSES_PASS"
        super().__init__(graph, _const_perm_opt_transposes_fun, name)


def _const_opt_transposes_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]

    if isinstance(t1.inputs[0], gs.Constant):
        t1.inputs[0].values = np.transpose(t1.inputs[0].values, t1.attrs['perm'])
        graph.deleteNode(t1)

    return graph


@contextagnostic
class TransposeConstOptPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'Transpose', name = 't1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_CONST_OPT_TRANSPOSES_PASS"
        super().__init__(graph, _const_opt_transposes_fun, name)


def _const_opt_reshape_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    reshape = matched_nodes[0]

    if isinstance(reshape.inputs[0], gs.Constant):
        reshape.inputs[0].values = reshape.inputs[0].values.reshape(reshape.inputs[1].values)
        graph.deleteNode(reshape)

    return graph


@contextagnostic
class ReshapeConstOptPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out'], op = 'Reshape', name = 'reshape')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_CONST_OPT_RESHAPE_PASS"
        super().__init__(graph, _const_opt_reshape_fun, name)


def _merge_reshape_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    reshape1 = matched_nodes[0]
    reshape2 = matched_nodes[1]

    graph.deleteNode(reshape1)

    graph.cleanup()

    return graph


@contextagnostic
class ReshapeMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['out1'], op = 'Reshape', name = 'reshape1')
        output = graph.layer(inputs = output, outputs = ['out2'], op = 'Reshape', name = 'reshape2')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_RESHAPE_PASS"
        super().__init__(graph, _merge_reshape_fun, name)


def _split_rqs_fun(graph: gs.Graph, match: Match, name: str, splitSet: List[str]):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    t1 = matched_nodes[0]

    if len(t1.outputs[0].outputs) <= 1:
        return graph

    outputOpNames = [node.op for node in t1.outputs[0].outputs]

    if not any([name in splitSet for name in outputOpNames]):
        return graph

    inputVars = t1.inputs
    inputNode = t1.inputs[0]

    originalNode = t1.outputs[0]

    userNodes = t1.outputs[0].outputs

    postSplitInputs = []
    for idx, var in enumerate(inputVars):
        if isinstance(var, gs.Variable):
            postSplitInput = var
        else:
            postSplitInput = gs.Constant(name = f"{t1.name}_split_{idx}", values = var.values.copy().reshape(-1,))
        postSplitInputs.append(postSplitInput)

    for idx, node in enumerate(originalNode.outputs.copy()):

        nodeName = node.name + f"_rqs"
        varName = node.name + f"_rqs_var"
        newOutput = gs.Variable(name = varName, dtype = np.float32, shape = t1.outputs[0].shape)

        RQSNode = gs.Node(name = nodeName,
                          op = "RequantShift",
                          inputs = postSplitInputs,
                          outputs = [newOutput],
                          attrs = t1.attrs)

        graph.nodes.append(RQSNode)

        newNodeInputs = []
        for _input in node.inputs:
            if _input != originalNode:
                newNodeInputs.append(_input)
            else:
                newNodeInputs.append(newOutput)

        node.inputs = newNodeInputs

    t1.outputs = []
    t1.inputs = []

    graph.cleanup().toposort()

    return graph


@contextagnostic
class RQSSplitPass(ReplaceSequentialPatternPass):

    splitSet = ["Add", "Concat"]

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['t1_out'], op = 'RequantShift', name = 't1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_SPLIT_RequantShift_PASS"
        super().__init__(graph, partial(_split_rqs_fun, splitSet = self.splitSet), name)


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
class AddRequantMergePass(ReplaceSequentialPatternPass):
    pass

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
class GEMMRequantMergePass(ReplaceSequentialPatternPass):

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
