# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Union

import onnx_graphsurgeon as gs


def graph_coloring(graph: gs.Graph, colors: List[str], frequency: Union[int, List[int]], color_attr: str):
    color_count = 0
    i_color = 0
    graph = graph.copy()

    if isinstance(frequency, int):
        frequency = [frequency] * len(colors)
    elif isinstance(frequency, list):
        assert len(frequency) == len(colors), "The length of frequency and colors does not match"

    for node in graph.nodes:
        if node.op == 'Constant':
            continue
        node.attrs[color_attr] = colors[i_color]
        for inputTensor in node.inputs:
            if not inputTensor.inputs:
                continue
            input = inputTensor.inputs[0]
            if isinstance(input, gs.Node) and input.op == 'Constant':
                input.attrs[color_attr] = colors[i_color]
        color_count += 1
        if color_count == frequency[i_color]:
            color_count = 0
            i_color = (i_color + 1) % len(colors)

    return graph
