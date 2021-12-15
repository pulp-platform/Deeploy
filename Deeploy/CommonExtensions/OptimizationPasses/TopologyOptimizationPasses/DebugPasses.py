# ----------------------------------------------------------------------
#
# File: DebugPasses.py
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
from functools import partial
from typing import Literal

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


@contextagnostic
class EmulateCMSISRequantPass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['output0'], op = 'RequantShift', name = 'rqs1')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_EMULATE_CMSIS_REQUANT_PASS"
        super().__init__(graph, _convert_requant_to_cmsis_fun, name)


def _convert_requant_to_cmsis_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    rqs = matched_nodes[0]

    # Make sure pass is only applied once
    if 'Emulate_CMSIS_RequantShift' in rqs.attrs:
        return graph

    # WIESEP: Because CMSIS performs add-multiply-divide and we normally do multiply-add-divide
    #         we can emulate the same behavior by rounding the MUL value
    rqs.inputs[-1].values = np.round(copy.deepcopy(rqs.inputs[-1].values) /
                                     (rqs.inputs[-2].values + 1e-3)) * rqs.inputs[-2].values
    rqs.attrs['emulate_CMSIS_requantShift'] = True

    return graph


def _print_fun(graph: gs.Graph, match: Match, name: str, position: Literal["before", "after"] = "before"):
    assert position in ["before", "after"], f"'{position}' is not a valid position for the print node!"

    matched_nodes = [m for k, m in match.nodes_map.items()]

    node = matched_nodes[0]
    name += '_' + node.name

    if position == 'before' and "PRINT" not in node.inputs[0].name:
        newNodeInput = gs.Variable(name + '_input', dtype = np.float32, shape = node.inputs[0].shape)
        newPrintNode = gs.Node(op = 'DebugPrint',
                               name = 'DebugPrint_' + node.name + '_input',
                               inputs = [node.inputs[0]],
                               outputs = [newNodeInput])

        node.inputs[0] = newNodeInput

        graph.nodes.append(newPrintNode)
        graph.cleanup().toposort()

    if position == 'after' and "PRINT" not in node.outputs[0].name:
        newNodeOutput = gs.Variable(name + '_output', dtype = np.float32, shape = node.outputs[0].shape)
        newPrintNode = gs.Node(op = 'DebugPrint',
                               name = 'DebugPrint_' + node.name + '_output',
                               inputs = [newNodeOutput],
                               outputs = [node.outputs[0]])

        node.outputs[0] = newNodeOutput

        graph.nodes.append(newPrintNode)
        graph.cleanup().toposort()

    return graph


@contextagnostic
class DebugPrintPass(ReplaceSequentialPatternPass):

    def __init__(self, op_regex: str, position = 'before'):

        if op_regex == "":
            raise ValueError('Operator not set!')
        if position not in ['before', 'after']:
            ValueError(f'Invalid position "{position}"!')

        pattern = gs.Graph()
        _input = gs.Variable(name = 'input_0')
        output = pattern.layer(inputs = [_input], outputs = ['output0'], op = op_regex)
        pattern.outputs.append(output)
        pattern.inputs = [_input]

        name = "_DEBUG_PRINT_PASS"
        super().__init__(pattern, partial(_print_fun, position = position), name, NonBranchingMatcher(regex_op = True))


def _merge_print_fun(graph: gs.Graph, match: Match, name: str):
    matched_nodes = [m for k, m in match.nodes_map.items()]
    d1 = matched_nodes[0]
    d2 = matched_nodes[1]

    _inputs = list(d1.inputs)
    _outputs = list(d2.outputs)

    newPrintNode = gs.Node(op = 'DebugPrint', name = name)
    graph.replaceInsertNode(_inputs, _outputs, newPrintNode)

    graph.cleanup().toposort()
    return graph


@contextagnostic
class DebugPrintMergePass(ReplaceSequentialPatternPass):

    def __init__(self):
        graph = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = graph.layer(inputs = [_input], outputs = ['d1_out'], op = 'DebugPrint', name = 'd1')
        output = graph.layer(inputs = output, outputs = ['d2_out'], op = 'DebugPrint', name = 'd2')
        graph.outputs.append(output)
        graph.inputs.append(_input)

        name = "_MERGE_DEBUG_PRINT_PASS"
        super().__init__(graph, _merge_print_fun, name)
