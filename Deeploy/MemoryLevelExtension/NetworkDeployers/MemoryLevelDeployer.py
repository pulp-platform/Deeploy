# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, DeploymentEngine, DeploymentPlatform, \
    NetworkContext, NetworkDeployer, NetworkOptimizationPass, NetworkOptimizer, ONNXLayer, Schedule, StructBuffer, \
    TopologyOptimizer, TransientBuffer, VariableBuffer, _NoVerbosity
from Deeploy.Logging import DEFAULT_LOGGER as log
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

        newCtxt, parsePass = super()._parseNode(node, ctxt, default_channels_first)

        if not parsePass:
            return newCtxt, False

        newCtxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(newCtxt, self.graph)

        return newCtxt, parsePass

    def bind(self):

        ret = super().bind()
        if not ret:
            return False

        log.info("- Perform Memory Level Annotation")
        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)

    def _printMemorySummary(self):
        log.info("Memory Usage Report:")
        log.info(f"{'Level':<22} {'Capacity (bytes)':>16}   {'Total':>8}   {'(Static + Dynamic)':<21} {'Usage':<6}")
        log.info("-" * 80)

        for level, dynamicSize in self.worstCaseBufferSize.items():
            staticSize = 0
            for _buffer in self.ctxt.globalObjects.values():
                # We do not count structs for now, since they are not properly modeled
                if isinstance(_buffer, ConstantBuffer) and _buffer._deploy and _buffer._memoryLevel == level:
                    staticSize += int((np.prod(_buffer.shape) * _buffer._type.referencedType.typeWidth // 8))

            capacity = self.Platform.memoryHierarchy.memoryLevels[level].size
            total = staticSize + dynamicSize

            log.info(f"{level:<22} {capacity:16,}   {total:8,d}   "
                     f"({staticSize:6,d} + {dynamicSize:7,d})  "
                     f"({total / capacity * 100:5.1f}%)")


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

        log.info("- Perform Memory Level Annotation")
        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)

    def _printMemorySummary(self):
        log.info("Memory Usage Report:")
        log.info(f"{'Level':<22} {'Capacity (bytes)':>16}   {'Total':>8}   {'(Static + Dynamic)':<21} {'Usage':<6}")
        log.info("-" * 80)

        for level, dynamicSize in self.worstCaseBufferSize.items():
            staticSize = 0
            for _buffer in self.ctxt.globalObjects.values():
                # We do not count structs for now, since they are not properly modeled
                if isinstance(_buffer, ConstantBuffer) and _buffer._deploy and _buffer._memoryLevel == level:
                    staticSize += int((np.prod(_buffer.shape) * _buffer._type.referencedType.typeWidth // 8))

            capacity = self.Platform.memoryHierarchy.memoryLevels[level].size
            total = staticSize + dynamicSize

            log.info(f"{level:<22} {capacity:16,}   {total:8,d}   "
                     f"({staticSize:6,d} + {dynamicSize:7,d})  "
                     f"({total / capacity * 100:5.1f}%)")


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

        log.info("- Perform Memory Level Annotation")
        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)

    def _printMemorySummary(self):
        log.info("Memory Usage Report:")
        log.info(f"{'Level':<22} {'Capacity (bytes)':>16}   {'Total':>8}   {'(Static + Dynamic)':<21} {'Usage':<6}")
        log.info("-" * 80)

        for level, dynamicSize in self.worstCaseBufferSize.items():
            staticSize = 0
            for _buffer in self.ctxt.globalObjects.values():
                # We do not count structs for now, since they are not properly modeled
                if isinstance(_buffer, ConstantBuffer) and _buffer._deploy and _buffer._memoryLevel == level:
                    staticSize += int((np.prod(_buffer.shape) * _buffer._type.referencedType.typeWidth // 8))

            capacity = self.Platform.memoryHierarchy.memoryLevels[level].size
            total = staticSize + dynamicSize

            log.info(f"{level:<22} {capacity:16,}   {total:8,d}   "
                     f"({staticSize:6,d} + {dynamicSize:7,d})  "
                     f"({total / capacity * 100:5.1f}%)")
