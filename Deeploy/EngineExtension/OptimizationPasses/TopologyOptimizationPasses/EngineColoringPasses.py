# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Optional, Tuple

import onnx_graphsurgeon as gs

from Deeploy.CommonExtensions.OptimizationPasses.Matchers import Match, SubgraphMatcher
from Deeploy.CommonExtensions.OptimizationPasses.PassClasses import ReplaceSequentialPatternPass, contextagnostic
from Deeploy.DeeployTypes import DeploymentEngine, TopologyOptimizationPass


class EngineMapper:

    def __init__(self, engineDict: Dict[str, DeploymentEngine]) -> None:
        self.engineDict = engineDict

    # Override for a different allocation strategy
    def mapNodeToEngine(self, node: gs.Node, graph: gs.Graph) -> Optional[DeploymentEngine]:
        _ = graph
        for engine in self.engineDict.values():
            if engine.canExecute(node):
                return engine
        return None


class EngineColoringPass(TopologyOptimizationPass):

    def __init__(self, engineMapper: EngineMapper):
        super().__init__()
        self.engineMapper = engineMapper

    def apply(self, graph: gs.Graph) -> Tuple[gs.Graph]:
        for node in filter(lambda node: "engine" not in node.attrs, graph.nodes):
            engine = self.engineMapper.mapNodeToEngine(node, graph)
            if engine is not None:
                node.attrs["engine"] = engine.name
        return graph


def _engine_discoloration_fun(graph: gs.Graph, match: Match, name: str):
    _ = name
    colored_matched_nodes = filter(lambda node: "engine" in node.attrs, match.nodes_map.values())
    for node in colored_matched_nodes:
        del node.attrs["engine"]
    return graph


@contextagnostic
class EngineDiscolorationPass(ReplaceSequentialPatternPass):

    def __init__(self, pattern: gs.Graph, name: str, matcher: Optional[SubgraphMatcher] = None, **kwargs):
        super().__init__(pattern, _engine_discoloration_fun, name, matcher, **kwargs)
