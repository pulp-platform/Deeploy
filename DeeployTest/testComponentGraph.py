# ----------------------------------------------------------------------
#
# File: testComponentGraph.py
#
# Last edited: 10.10.2023.
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
#   - Luka Macan, University of Bologna
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

import os

import onnx
import onnx_graphsurgeon as gs
from testUtils.graphColoring import graph_coloring

from Deeploy.ComponentGraph import extractComponentGraph, extractComponentsFromComponentGraph

if __name__ == "__main__":
    test_dir = "Tests/WaveFormer"
    colors = ["red", "green", "blue", "yellow"]
    component_color = "red"
    color_attr = "color"
    color_frequency = 10

    model = onnx.load(os.path.join(test_dir, "network.onnx"))
    graph = gs.import_onnx(model).toposort()

    # Color the graph randomly
    graph = graph_coloring(graph, colors = colors, frequency = color_frequency, color_attr = color_attr)

    # Color a few output nodes
    N_OUTPUT_NODES = 5
    first_output_node = graph.outputs[0].inputs[0]
    next_nodes = [first_output_node]
    for i in range(N_OUTPUT_NODES):
        node = next_nodes.pop()
        node.attrs[color_attr] = component_color
        next_nodes += [tensor.inputs[0] for tensor in node.inputs if isinstance(tensor, gs.Variable)]

    # Check that all the nodes have been colored
    for node in graph.nodes:
        assert color_attr in node.attrs

    componentGraph = extractComponentGraph(graph, lambda node: node.attrs[color_attr] == component_color)

    model = gs.export_onnx(componentGraph)
    onnx.save_model(model, "component_graph.onnx")

    # Check that all the nodes in the components are of the component_color
    for node in componentGraph.nodes:
        assert node.attrs[
            color_attr] == component_color, f"Node {node.name} is not of {component_color} but {node.attrs['color']}"

    # Check that all the component_color nodes from the original graph exist in the components
    for node in graph.nodes:
        if node.attrs[color_attr] == component_color:
            assert any(node.name == componentNode.name for componentNode in componentGraph.nodes
                      ), f"Node {node.name} of color {component_color} does not exist in any of the components"

    # Check for duplicates in the inputs
    inputNames = [tensor.name for tensor in componentGraph.inputs]
    assert len(inputNames) == len(set(inputNames))

    # Check for duplicates in the outputs
    outputNames = [tensor.name for tensor in componentGraph.outputs]
    assert len(outputNames) == len(set(outputNames))

    components = extractComponentsFromComponentGraph(componentGraph)

    componentNodes = []
    for component in components:
        componentNodes += list(component.nodes)

    # Check components contain all the nodes from the componentGraph
    for node in componentGraph.nodes:
        assert any(node.name == componentNode.name
                   for componentNode in componentNodes), f"Node {node.name} is not present in any of the components"

    # Check if there are nodes that are in multiple components
    nodeNames = [node.name for node in componentNodes]
    assert len(nodeNames) == len(set(nodeNames))

    print("Test passed")
