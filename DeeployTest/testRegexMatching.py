# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import re

import onnx
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, NonBranchingMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic
from Deeploy.DeeployTypes import TopologyOptimizer

# Match any operation that contains conv in it's name
test_regex = r'.*[Cc]onv.*'
test_op_name = 'TestConv'


def _rename_conv_to_test_conv(graph: gs.Graph, match: Match, name: str):
    matched_nodes = list(match.nodes_map.values())
    assert len(matched_nodes) == 1
    conv = matched_nodes[0]
    testConv = gs.Node(op = test_op_name, name = name, attrs = {**conv.attrs})
    graph.replaceInsertNode(conv.inputs, conv.outputs, testConv)
    return graph


# Match all nodes with 'conv' in their name and add a `test` attribute
@contextagnostic
class ConvTestPass(ReplaceSequentialPatternPass):

    def __init__(self):
        pattern = gs.Graph()
        _input = gs.Variable(name = 'input_1')
        output = pattern.layer(inputs = [_input], outputs = ['conv_out'], op = test_regex, name = 'conv1')
        pattern.outputs.append(output)
        pattern.inputs.append(_input)

        name = "_CONV_TEST_PASS"
        super().__init__(pattern, _rename_conv_to_test_conv, name, NonBranchingMatcher(regex_op = True))


if __name__ == "__main__":
    optimizer = TopologyOptimizer([ConvTestPass()])
    model = onnx.load_model('Tests/simpleCNN/network.onnx')
    graph = gs.import_onnx(model)

    match_count = 0

    for node in graph.nodes:
        if re.match(test_regex, node.op):
            match_count += 1

    optimized_graph = optimizer.optimize(graph)

    test_op_name_count = 0

    for node in optimized_graph.nodes:
        if node.op == test_op_name:
            test_op_name_count += 1

    assert match_count == test_op_name_count, "Didn't match all the operations."

    print("Test passed")
