# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from types import MappingProxyType
from typing import Any, Callable, Dict, List, Tuple, Type, Union

import onnx_graphsurgeon as gs

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.CommonExtensions.NetworkDeployers.NetworkDeployerWrapper import NetworkDeployerWrapper
from Deeploy.CommonExtensions.NetworkDeployers.SignPropDeployer import SignPropDeployer
from Deeploy.DeeployTypes import CodeGenVerbosity, ConstantBuffer, DeploymentEngine, DeploymentPlatform, \
    NetworkContext, NetworkDeployer, NetworkOptimizationPass, NetworkOptimizer, Schedule, StructBuffer, \
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


class MemorySummaryMixin:

    def _printMemorySummary(self):
        log.info("")
        log.info("Memory Usage Report:")
        log.info(f"  {'Level':<14} {'Capacity (bytes)':>10} {'Total':>10} (    Static + Dynamic   ) (Usage )")
        log.info("  " + "-" * 78)

        for level, dynamicSize in self.worstCaseBufferSize.items():
            staticSize = 0
            for _buffer in self.ctxt.globalObjects.values():
                # We do not count structs for now, since they are not properly modeled
                if isinstance(_buffer, ConstantBuffer) and getattr(_buffer, "_deploy", False):
                    if (hasattr(_buffer, "_memoryLevel") and _buffer._memoryLevel == level) or level in ("None", None):
                        staticSize += _buffer.sizeInBytes

            total = staticSize + dynamicSize
            memLevels = self.Platform.memoryHierarchy.memoryLevels
            memLevel = memLevels.get(level, None)
            if memLevel is None or getattr(memLevel, "size", None) is None:
                log.info(f"  {str(level):<20} {'N/A':>10} {total:10,d} "
                         f"({staticSize:10,d} + {dynamicSize:10,d}) "
                         f"({'N/A':>5})")
            else:
                capacity = memLevel.size
                log.info(f"  {str(level):<20} {capacity:10,} {total:10,d} "
                         f"({staticSize:10,d} + {dynamicSize:10,d}) "
                         f"({total / capacity * 100:5.1f}%)")


class MemoryLevelAwareDeployer(NetworkDeployer, MemorySummaryMixin):

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

    def bind(self):
        log.info("- Perform Memory Level Annotation")
        # LMACAN: Annotate before bind because during binding (specifically alignToContext) templates
        #         may expect the memoryLevel annotation already.
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        ret = super().bind()
        if not ret:
            return False

        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)


class MemoryLevelAwareSignPropDeployer(SignPropDeployer, MemorySummaryMixin):

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

    def bind(self):
        log.info("- Perform Memory Level Annotation")
        # LMACAN: Annotate before bind because during binding (specifically alignToContext) templates
        #         may expect the memoryLevel annotation already.
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        ret = super().bind()
        if not ret:
            return False

        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)


class MemoryDeployerWrapper(NetworkDeployerWrapper, MemorySummaryMixin):

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

    def bind(self):
        log.info("- Perform Memory Level Annotation")
        # LMACAN: Annotate before bind because during binding (specifically alignToContext) templates
        #         may expect the memoryLevel annotation already.
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        ret = super().bind()
        if not ret:
            return False

        # SCHEREMO: There might be hoisting; reassign memoryLevel preferences
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)

        return ret

    def codeTransform(self, verbose: CodeGenVerbosity = _NoVerbosity):
        self.ctxt, self.graph = self.memoryLevelAnnotationOptimizer.optimize(self.ctxt, self.graph)
        super().codeTransform(verbose)
