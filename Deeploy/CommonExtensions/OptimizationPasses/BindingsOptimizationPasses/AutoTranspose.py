# ----------------------------------------------------------------------
#
# File: AutoTranspose.py
#
# Last edited: 20.11.2023
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

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.BindingsOptimizationPasses.bindingUtils import bypassNode, \
    editAttribute
from Deeploy.CommonExtensions.OptimizationPasses.BindingsOptimizationPasses.PassClasses import bindingaware
from Deeploy.CommonExtensions.OptimizationPasses.Matchers import BranchingMatcher, Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, SequentialPass
from Deeploy.DeeployTypes import NetworkContext


def _merge_transposeRequantInputs_fun(ctxt: NetworkContext, layerBinding, match: Match, name: str):

    nodes_map = match.nodes_map
    in1Node = nodes_map['any1']
    rqNode = nodes_map['rqsOut']

    mergeCondition = lambda node: all([
        node.op == "Transpose", not "bypass" in node.attrs.keys(),
        len(node.outputs[0].outputs) == 1, rqNode.attrs["targetMemoryLevelMap"][rqNode.outputs[0].name] == "L1",
        ctxt.lookup(node.outputs[0].name)._memoryLevel != "L1"
    ])

    # if rqNode.op == "RequantizedAdd":
    #     return ctxt, layerBinding

    if not hasattr(ctxt.lookup(in1Node.outputs[0].name), "_memoryLevel"):
        return ctxt, layerBinding

    orderedInputNames = [inp.name for inp in rqNode.inputs]

    if mergeCondition(in1Node):

        idx = orderedInputNames.index(in1Node.outputs[0].name)

        editAttribute(layerBinding, rqNode, f"in{idx}_perm", in1Node.attrs["perm"])
        editAttribute(layerBinding, in1Node, "bypass", 1)
        ctxt, layerBinding = bypassNode(ctxt, layerBinding, in1Node)

    if 'any2' not in nodes_map.keys():
        return ctxt, layerBinding

    in2Node = nodes_map['any2']

    if not hasattr(ctxt.lookup(in2Node.outputs[0].name), "_memoryLevel"):
        return ctxt, layerBinding

    if mergeCondition(in2Node):
        idx = orderedInputNames.index(in2Node.outputs[0].name)

        editAttribute(layerBinding, rqNode, f"in{idx}_perm", in2Node.attrs["perm"])
        editAttribute(layerBinding, in2Node, "bypass", 1)
        ctxt, layerBinding = bypassNode(ctxt, layerBinding, in2Node)

    return ctxt, layerBinding


def _merge_transposeRequantOutputs_fun(ctxt: NetworkContext, layerBinding, match: Match, name: str):

    nodes_map = match.nodes_map

    outNode = nodes_map['anyOut']
    rqNode = nodes_map['rqs']

    if not hasattr(ctxt.lookup(rqNode.outputs[0].name), "_memoryLevel"):
        return ctxt, layerBinding

    mergeCondition = lambda node: all([
        node.op == "Transpose", not "bypass" in node.attrs.keys(),
        len(rqNode.outputs[0].outputs) == 1, rqNode.attrs["targetMemoryLevelMap"][rqNode.outputs[0].name] == "L1",
        ctxt.lookup(rqNode.outputs[0].name)._memoryLevel != "L1"
    ])

    if mergeCondition(outNode):
        editAttribute(layerBinding, rqNode, "out_perm", outNode.attrs["perm"])
        editAttribute(layerBinding, outNode, "bypass", 1)

    return ctxt, layerBinding


@bindingaware
class _CAPass(ReplaceSequentialPatternPass):
    pass


@bindingaware
class AutoTransposeMergeOutputsPass(SequentialPass):

    def _buildSingleVarGraph(self) -> gs.Graph:

        _input1 = gs.Variable(name = 'input_1')

        _rqs = gs.Variable(name = 'rqsVar')
        _rqsOut = gs.Variable(name = 'rqsOutput')

        output = gs.Node(inputs = [_input1], outputs = [_rqs], op = r'Requantized.*', name = 'rqs')
        rqsOut = gs.Node(inputs = [_rqs], outputs = [_rqsOut], op = r'.*', name = 'anyOut')

        graph = gs.Graph(nodes = [output, rqsOut], inputs = [_input1], outputs = [_rqsOut]).cleanup()

        return graph

    def _buildDualVarGraph(self) -> gs.Graph:

        _input1 = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')

        _rqs = gs.Variable(name = 'rqsVar')
        _rqsOut = gs.Variable(name = 'rqsOutput')

        output = gs.Node(inputs = [_input1, _input2], outputs = [_rqs], op = r'Requantized.*', name = 'rqs')
        rqsOut = gs.Node(inputs = [_rqs], outputs = [_rqsOut], op = r'.*', name = 'anyOut')

        graph = gs.Graph(nodes = [output, rqsOut], inputs = [_input1, _input2], outputs = [_rqsOut]).cleanup()

        return graph

    def __init__(self):

        pass1 = _CAPass(self._buildSingleVarGraph(),
                        replacement_fn = _merge_transposeRequantOutputs_fun,
                        name = "_MERGE_TransposeRQ_PASS",
                        matcher = NonBranchingMatcher(regex_op = True))

        pass2 = _CAPass(self._buildDualVarGraph(),
                        replacement_fn = _merge_transposeRequantOutputs_fun,
                        name = "_MERGE_TransposeRQ_PASS",
                        matcher = BranchingMatcher(regex_op = True))

        super().__init__(pass1, pass2)


@bindingaware
class AutoTransposeMergeInputsPass(SequentialPass):

    def _buildSingleVarGraph(self) -> gs.Graph:

        _input1 = gs.Variable(name = 'input_1')
        _rqIn1 = gs.Variable(name = 'rqIn1')

        _rqs = gs.Variable(name = 'rqs')

        anyIn1 = gs.Node(inputs = [_input1], outputs = [_rqIn1], op = r'.*', name = 'any1')
        output = gs.Node(inputs = [_rqIn1], outputs = [_rqs], op = r'Requantized.*', name = 'rqsOut')

        graph = gs.Graph(nodes = [anyIn1, output], inputs = [_input1], outputs = [_rqs])

        return graph

    def _buildDualVarGraph(self) -> gs.Graph:

        _input1 = gs.Variable(name = 'input_1')
        _input2 = gs.Variable(name = 'input_2')

        _rqIn1 = gs.Variable(name = 'rqIn1')
        _rqIn2 = gs.Variable(name = 'rqIn2')

        _rqs = gs.Variable(name = 'rqs')
        _rqsOut = gs.Variable(name = 'rqs')

        anyIn1 = gs.Node(inputs = [_input1], outputs = [_rqIn1], op = r'.*', name = 'any1')
        anyIn2 = gs.Node(inputs = [_input2], outputs = [_rqIn2], op = r'.*', name = 'any2')

        output = gs.Node(inputs = [_rqIn1, _rqIn2], outputs = [_rqs], op = r'Requantized.*', name = 'rqsOut')

        graph = gs.Graph(nodes = [anyIn1, anyIn2, output], inputs = [_input1, _input2], outputs = [_rqs])

        return graph

    def __init__(self):

        pass1 = _CAPass(self._buildSingleVarGraph(),
                        replacement_fn = _merge_transposeRequantInputs_fun,
                        name = "_MERGE_TransposeRQ_PASS",
                        matcher = NonBranchingMatcher(regex_op = True))

        pass2 = _CAPass(self._buildDualVarGraph(),
                        replacement_fn = _merge_transposeRequantInputs_fun,
                        name = "_MERGE_TransposeRQ_PASS",
                        matcher = BranchingMatcher(regex_op = True))

        super().__init__(pass1, pass2)


@bindingaware
class AutoTransposeMergePass(SequentialPass):

    def __init__(self):

        pass1 = AutoTransposeMergeInputsPass()
        # SCHEREMO: Not sure if PULP supports DMA'ing outputs transposed
        #pass2 = AutoTransposeMergeOutputsPass()

        super().__init__(pass1)
