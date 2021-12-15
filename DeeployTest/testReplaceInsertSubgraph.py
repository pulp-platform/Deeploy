# ----------------------------------------------------------------------
#
# File: testReplaceInsertSubgraph.py
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

from Deeploy.DeeployTypes import TopologyOptimizer
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.PULPPasses import PULPConvRequantMergePass

if __name__ == "__main__":
    test_dir = "Tests/simpleRegression"

    model = onnx.load(os.path.join(test_dir, "network.onnx"))
    graph = gs.import_onnx(model).toposort()
    graph_len = len(graph.nodes)

    subgraph = graph.copy()
    # Make the subgraph the first 3 nodes
    subgraph_len = 3
    subgraph.outputs = [subgraph.nodes[subgraph_len - 1].outputs[0]]
    subgraph.cleanup()

    assert len(subgraph.nodes) == subgraph_len
    assert len(graph.nodes) == graph_len

    optimizer = TopologyOptimizer([PULPConvRequantMergePass()])

    subgraph = optimizer.optimize(subgraph)
    graph.replaceInsertSubgraph(subgraph)
    graph = graph.cleanup().toposort()

    assert graph.nodes[
        0].op == "RequantizedConv", f"First 2 nodes should have been replaced with the RequantizedConv node. Got {graph.nodes[0].op}"

    non_optimized = [2, 5, 8]
    for i in non_optimized:
        assert graph.nodes[i].op == "Conv" and graph.nodes[
            i +
            1].op == "RequantShift", f"Nodes outside of the subgraph shouldn't have been optimized. Failed on nodes {i}:{graph.nodes[i].op} and {i+1}:{graph.nodes[i+1].op}"

    print("Test passed")
