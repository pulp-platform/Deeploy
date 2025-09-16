# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Callable, Dict, Type, Union

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.DeeployTypes import DeploymentPlatform, NetworkDeployer, ONNXLayer, Schedule, TopologyOptimizer
from Deeploy.EngineExtension.OptimizationPasses.TopologyOptimizationPasses.EngineColoringPasses import \
    EngineColoringPass, EngineMapper


class EngineColoringDeployer(NetworkDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: DeploymentPlatform,
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable[[gs.Graph], Schedule] = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 engineMapperCls: Type[EngineMapper] = EngineMapper):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir)
        self._initEngineColoringDeployer(engineMapperCls)

    def _initEngineColoringDeployer(self, engineMapperCls: Type[EngineMapper]):
        self.engineDict = {engine.name: engine for engine in self.Platform.engines}
        engineMapper = engineMapperCls(self.engineDict)
        engineColoringPass = EngineColoringPass(engineMapper)
        loweringPasses = [engineColoringPass]
        for _pass in self.loweringOptimizer.passes:
            loweringPasses.append(_pass)
            loweringPasses.append(engineColoringPass)
        self.loweringOptimizer.passes = loweringPasses

    def lower(self, graph: gs.Graph) -> gs.Graph:
        graph = super().lower(graph)
        uncoloredNodes = [node for node in graph.nodes if "engine" not in node.attrs]
        uncoloredOperations = set(node.op for node in uncoloredNodes)
        assert len(
            uncoloredNodes
        ) == 0, f"Missing engine color for nodes {[node.name for node in uncoloredNodes]} with operations {uncoloredOperations}"
        return graph

    def _mapNode(self, node: gs.Node) -> Union[ONNXLayer, Any]:
        assert "engine" in node.attrs, f"Node {node.name} doesn't have an engine color."
        engineName = node.attrs["engine"]
        assert isinstance(engineName, str) and engineName in self.engineDict, \
            f"Node {node.name} has an invalid engine {engineName} assigned."
        engine = self.engineDict[engineName]
        assert node.op in engine.Mapping, f"No mapping found for {node.op} in engine {engine.name}"
        return engine.Mapping[node.op](node)


class EngineColoringDeployerWrapper(EngineColoringDeployer, NetworkDeployerWrapper):

    def __init__(self, deployer: NetworkDeployer, engineMapperCls: Type[EngineMapper] = EngineMapper) -> None:
        NetworkDeployerWrapper.__init__(self, deployer)
        self._initEngineColoringDeployer(engineMapperCls)
