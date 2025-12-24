# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import os

import onnx
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import TopologyOptimizer
from Deeploy.OptimizationPasses.TopologyOptimizationPasses.PULPPasses import PULPConvRequantMergePass

if __name__ == "__main__":
    test_dir = "Tests/Models/CNN_Linear2"

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
