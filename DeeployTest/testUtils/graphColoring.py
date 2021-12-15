# ----------------------------------------------------------------------
#
# File: graphColoring.py
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
