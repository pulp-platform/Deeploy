# ----------------------------------------------------------------------
#
# File: graphDebug.py
#
# Last edited: 23.05.2023
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

from typing import Dict, Tuple

import numpy as np
import onnx_graphsurgeon as gs

from .graphDiffUtils import DiffTree, createParentDiffNode, listDiff, nodeDiff, tensorDiff


def generateDebugConfig(test_inputs_files, test_outputs_files, activations_files,
                        graph: gs.Graph) -> Tuple[Dict, Dict, gs.Graph]:

    test_inputs = [test_inputs_files[x].reshape(-1).astype(np.int64) for x in test_inputs_files.files]
    test_outputs = [test_outputs_files[x].reshape(-1).astype(np.int64) for x in test_outputs_files.files]

    import IPython
    IPython.embed()

    return test_inputs, test_outputs, graph


def graphDiff(graph: gs.Graph, other: gs.Graph) -> DiffTree:
    graph = graph.toposort()
    other = other.toposort()
    diffs = []
    diffs.append(listDiff(graph.nodes, other.nodes, "nodes", nodeDiff))
    diffs.append(listDiff(graph.inputs, other.inputs, "inputs", tensorDiff))
    diffs.append(listDiff(graph.outputs, other.outputs, "outputs", tensorDiff))
    root = createParentDiffNode(graph, other, graph.name, diffs)
    return DiffTree(root)
