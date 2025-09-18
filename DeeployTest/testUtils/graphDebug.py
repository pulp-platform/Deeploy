# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

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
