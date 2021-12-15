# ----------------------------------------------------------------------
#
# File: MemoryLevelAnnotation.py
#
# Last edited: 04.05.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Author:
# Moritz Scherer, ETH Zurich
# Victor Jung, ETH Zurich
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

from types import MappingProxyType
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, DeploymentEngine, DeploymentPlatform, \
    NetworkContext, NetworkDeployer, NetworkOptimizationPass, NetworkOptimizer, ONNXLayer, Schedule, StructBuffer, \
    TopologyOptimizer, TransientBuffer, VariableBuffer, _NoVerbosity
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.OptimizationPasses.MemoryLevelAnnotationPasses import AnnotateDefaultMemoryLevel


class MemoryPlatform(DeploymentPlatform):

    def __init__(self, memoryHierarchy: MemoryHierarchy, defaultTargetMemoryLevel: MemoryLevel,
                 engines: List[DeploymentEngine], variableBuffer: Type[VariableBuffer],
                 constantBuffer: Type[ConstantBuffer], structBuffer: Type[StructBuffer],
                 transientBuffer: Type[TransientBuffer]) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
        self.memoryHierarchy = memoryHierarchy
        self.defaultTargetMemoryLevel = defaultTargetMemoryLevel

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        _, _, _ = node, tensorName, ctxt
        return self.defaultTargetMemoryLevel.name


class DeploymentPlatformWrapper(DeploymentPlatform):

    def __init__(self, platform: DeploymentPlatform):
        super().__setattr__("_innerObject", platform)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._innerObject, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self._innerObject, name):
            setattr(self._innerObject, name, value)
        else:
            super().__setattr__(name, value)


class MemoryPlatformWrapper(DeploymentPlatformWrapper):

    def __init__(self, platform: DeploymentPlatform, memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel):
        super().__init__(platform)
        self.memoryHierarchy = memoryHierarchy
        self.defaultTargetMemoryLevel = defaultTargetMemoryLevel

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        _, _, _ = node, tensorName, ctxt
        return self.defaultTargetMemoryLevel.name


class TargetMemoryLevelMapping:

    def __init__(self, graph: gs.Graph, platform: Union[MemoryPlatform, MemoryPlatformWrapper],
                 ctxt: NetworkContext) -> None:
        mapping: Dict[Tuple[str, str], str] = {}
        for node in graph.nodes:
            for tensor in node.inputs + node.outputs:
                mapping[node.name, tensor.name] = platform.getTargetMemoryLevel(node, tensor.name, ctxt)
        self._mapping = MappingProxyType(mapping)

    def lookup(self, nodeName: str, tensorName: str) -> str:
        return self._mapping[nodeName, tensorName]


class MemoryLevelAwareDeployer(NetworkDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: Union[MemoryPlatform, MemoryPlatformWrapper],
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable[[gs.Graph], Schedule] = lambda graph: list(graph.nodes),
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 memoryLevelAnnotationPasses: List[NetworkOptimizationPass] = []):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir)
        if len(memoryLevelAnnotationPasses) == 0:
            memoryLevelAnnotationPasses.append(AnnotateDefaultMemoryLevel(self.Platform.memoryHierarchy))
        self.memoryLevelAnnotationOptimizer = NetworkOptimizer(memoryLevelAnnotationPasses)

    def getTargetMemoryLevelMapping(self) -> TargetMemoryLevelMapping:
        assert isinstance(self.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(self.Platform).__name__}"
        return TargetMemoryLevelMapping(self.graph, self.Platform, self.ctxt)

    def _parseNode(self, node: ONNXLayer, ctxt: NetworkContext,
                   default_channels_first: bool) -> Tuple[NetworkContext, bool]:

        newCtxt, parsePass = node.parse(ctxt.copy(), default_channels_first)

        if not parsePass:
            return ctxt, False

        newCtxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(newCtxt, self.graph)
        newCtxt, LayerBindSuccess = node.typeCheck(newCtxt)

        if not LayerBindSuccess:
            return ctxt, False

        return newCtxt, True

    def bind(self):

        ret = super().bind()
        if not ret:
            return False

        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)


class MemoryLevelAwareSignPropDeployer(SignPropDeployer):

    def __init__(self,
                 graph: gs.Graph,
                 deploymentPlatform: Union[MemoryPlatform, MemoryPlatformWrapper],
                 inputTypes: Dict[str, Type[Pointer]],
                 loweringOptimizer: TopologyOptimizer,
                 scheduler: Callable = lambda x: x,
                 name: str = 'DeeployNetwork',
                 default_channels_first: bool = True,
                 deeployStateDir: str = "DeeployState",
                 inputOffsets: Dict[str, int] = {},
                 memoryLevelAnnotationPasses: List[NetworkOptimizationPass] = []):
        super().__init__(graph, deploymentPlatform, inputTypes, loweringOptimizer, scheduler, name,
                         default_channels_first, deeployStateDir, inputOffsets)
        if len(memoryLevelAnnotationPasses) == 0:
            memoryLevelAnnotationPasses.append(AnnotateDefaultMemoryLevel(self.Platform.memoryHierarchy))
        self.memoryLevelAnnotationOptimizer = NetworkOptimizer(memoryLevelAnnotationPasses)

    def getTargetMemoryLevelMapping(self) -> TargetMemoryLevelMapping:
        assert isinstance(self.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(self.Platform).__name__}"
        return TargetMemoryLevelMapping(self.graph, self.Platform, self.ctxt)

    def _parseNode(self, node: ONNXLayer, ctxt: NetworkContext,
                   default_channels_first: bool) -> Tuple[NetworkContext, bool]:

        newCtxt, parsePass = node.parse(ctxt.copy(), default_channels_first)

        if not parsePass:
            return ctxt, False

        newCtxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(newCtxt, self.graph)
        newCtxt, LayerBindSuccess = node.typeCheck(newCtxt)

        if not LayerBindSuccess:
            return ctxt, False

        return newCtxt, True

    def bind(self):

        ret = super().bind()
        if not ret:
            return False

        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)


class MemoryDeployerWrapper(NetworkDeployerWrapper):

    def __init__(self, deployer: NetworkDeployer, memoryLevelAnnotationPasses: List[NetworkOptimizationPass] = []):
        super().__init__(deployer)
        assert isinstance(deployer.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(deployer.Platform).__name__}"
        if len(memoryLevelAnnotationPasses) == 0:
            memoryLevelAnnotationPasses.append(AnnotateDefaultMemoryLevel(self.Platform.memoryHierarchy))
        self.memoryLevelAnnotationOptimizer = NetworkOptimizer(memoryLevelAnnotationPasses)

    def getTargetMemoryLevelMapping(self) -> TargetMemoryLevelMapping:
        assert isinstance(self.Platform, (MemoryPlatform, MemoryPlatformWrapper)), \
            f"Platform should be a MemoryPlatform or MemoryPlatformWrapper! Got {type(self.Platform).__name__}"
        return TargetMemoryLevelMapping(self.graph, self.Platform, self.ctxt)

    def _parseNode(self, node: ONNXLayer, ctxt: NetworkContext,
                   default_channels_first: bool) -> Tuple[NetworkContext, bool]:

        newCtxt, parsePass = node.parse(ctxt.copy(), default_channels_first)

        if not parsePass:
            return ctxt, False

        newCtxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(newCtxt, self.graph)
        newCtxt, LayerBindSuccess = node.typeCheck(newCtxt)

        if not LayerBindSuccess:
            return ctxt, False

        return newCtxt, True

    def bind(self):

        ret = super().bind()
        if not ret:
            return False

        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)
