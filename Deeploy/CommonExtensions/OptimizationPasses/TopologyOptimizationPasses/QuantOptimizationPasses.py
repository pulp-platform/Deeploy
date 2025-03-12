# ----------------------------------------------------------------------
#
# File: QuantOptimizationPasses.py
#
# Last edited: 07.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author: Federico Brancasi, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic


def _quant_pattern_fun(graph: gs.Graph, match: Match, name: str):
    # Get all nodes in the match
    matched_nodes = [m for k, m in match.nodes_map.items()]

    # The pattern should be: Div -> Add -> Round -> Clip
    # Extract each operation from the matched nodes
    div_node = matched_nodes[0]  # Div operation for scaling
    add_node = matched_nodes[1]  # Add operation for zero_point
    round_node = matched_nodes[2]  # Round operation
    clip_node = matched_nodes[3]  # Clip operation for clamping

    # Get input and output tensors
    input_tensor = div_node.inputs[0]
    output_tensor = clip_node.outputs[0]

    # Extract scale (from the second input of Div node)
    scale_input = div_node.inputs[1]
    scale_value = 1.0 / float(scale_input.values.item()) if hasattr(scale_input, 'values') else 1.0

    # Extract zero_point (from the second input of Add node)
    zero_point_input = add_node.inputs[1]
    zero_point_value = float(zero_point_input.values.item()) if hasattr(zero_point_input, 'values') else 0.0

    # Extract min and max values (from Clip node)
    min_value = clip_node.attrs.get('min')
    max_value = clip_node.attrs.get('max')

    # Determine bit_width and signed from min/max
    if min_value < 0:
        signed = True
        bit_width = int(np.log2(max_value - min_value + 1))
    else:
        signed = False
        bit_width = int(np.log2(max_value + 1))

    # Create a new Quant node with all attributes
    quant_attrs = {
        'scale': np.array([scale_value], dtype = np.float32),
        'zero_point': np.array([zero_point_value], dtype = np.float32),
        'bit_width': np.array([bit_width], dtype = np.int32),
        'signed': np.array([1 if signed else 0], dtype = np.int32),
        'min_val': np.array([min_value], dtype = np.int32),
        'max_val': np.array([max_value], dtype = np.int32)
    }

    # Create the new Quant node
    quant_node = gs.Node(op = 'Quant',
                         name = name + '_Quant',
                         inputs = [input_tensor],
                         outputs = [output_tensor],
                         attrs = quant_attrs)

    # Add the new node to the graph
    graph.nodes.append(quant_node)

    # Remove the old nodes
    for node in matched_nodes:
        node.inputs.clear()
        node.outputs.clear()
        graph.nodes.remove(node)

    return graph


@contextagnostic
class QuantPatternPass(ReplaceSequentialPatternPass):

    def __init__(self):
        # Define the pattern to match: Div -> Add -> Round -> Clip
        graph = gs.Graph()
        input_var = gs.Variable(name = 'input')
        scale_var = gs.Variable(name = 'scale')
        zero_point_var = gs.Variable(name = 'zero_point')

        # Create the pattern
        div_out = graph.layer(inputs = [input_var, scale_var], outputs = ['div_out'], op = 'Div', name = 'div')
        add_out = graph.layer(inputs = [div_out, zero_point_var], outputs = ['add_out'], op = 'Add', name = 'add')
        round_out = graph.layer(inputs = [add_out], outputs = ['round_out'], op = 'Round', name = 'round')
        clip_out = graph.layer(inputs = [round_out], outputs = ['clip_out'], op = 'Clip', name = 'clip')

        graph.outputs.append(clip_out)
        graph.inputs.append(input_var)

        name = "_QUANT_PATTERN_PASS"
        super().__init__(graph, _quant_pattern_fun, name)
