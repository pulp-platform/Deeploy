# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NetworkContext, NodeMapper, \
    NodeTemplate, StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform, MemoryPlatformWrapper
from Deeploy.Targets.Generic.Layers import AddLayer, SiLULayer
from Deeploy.Targets.Generic.Parsers import AddParser, SiLUParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.XDNA2.Bindings import XDNA2AddBindings, XDNA2SiLUBindings
from Deeploy.Targets.XDNA2.Tiler import XDNA2AddTilingReadyBindings, XDNA2SiLUTilingReadyBindings

# Standard mapper for non-tiled deployment
XDNA2AddMapper = NodeMapper(AddParser(), XDNA2AddBindings)
XDNA2SiLUMapper = NodeMapper(SiLUParser(), XDNA2SiLUBindings)

# Tiling-ready mapper for tiled deployment
XDNA2AddTilableMapper = NodeMapper(AddParser(), XDNA2AddTilingReadyBindings)
XDNA2SiLUTilableMapper = NodeMapper(SiLUParser(), XDNA2SiLUTilingReadyBindings)

# Standard mapping (used when tiling is disabled)
XDNA2Mapping = {
    'Add': AddLayer([XDNA2AddMapper]),
    'Silu': SiLULayer([XDNA2SiLUMapper]),
}

# Tiling-ready mapping (used when tiling is enabled)
XDNA2TilingMapping = {
    'Add': AddLayer([XDNA2AddTilableMapper]),
    'Silu': SiLULayer([XDNA2SiLUTilableMapper]),
}

# Buffer classes reuse Generic templates since XDNA2Deployer manages its own
# output format (MLIR + test headers) and these templates are never rendered.


class XDNA2VariableBuffer(VariableBuffer):
    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class XDNA2TransientBuffer(TransientBuffer):
    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class XDNA2ConstantBuffer(ConstantBuffer):
    initTemplate = AllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = AllocateTemplate.referenceGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.referenceGlobalTemplate


class XDNA2StructBuffer(StructBuffer):
    initTemplate = AllocateTemplate.referenceStructInitTemplate
    allocTemplate = AllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


# No topology optimization passes needed for the initial Add-only platform.
XDNA2Optimizer = TopologyOptimizer([], name = "XDNA2Optimizer")


class XDNA2Engine(DeploymentEngine):

    def __init__(self, name: str = "XDNA2", Mapping = XDNA2Mapping, initCode: str = "", includeList = None) -> None:
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)


class XDNA2AIECoreEngine(DeploymentEngine):
    """AIE core execution engine with L1 local memory as preferred memory level.

    The AIE core has 8KB of local memory (L1) for temporary buffers and computation.
    Data is transferred from L3 (shared memory) to L1 as needed.
    """

    def __init__(self,
                 name: str = "XDNA2_AIE_Core",
                 Mapping = XDNA2Mapping,
                 initCode: str = "",
                 includeList = None,
                 preferredMemoryLevel: str = "L1") -> None:
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)
        self.preferredMemoryLevel = preferredMemoryLevel


class XDNA2Platform(DeploymentPlatform):

    def __init__(self,
                 engines = None,
                 variableBuffer = XDNA2VariableBuffer,
                 constantBuffer = XDNA2ConstantBuffer,
                 structBuffer = XDNA2StructBuffer,
                 transientBuffer = XDNA2TransientBuffer):
        if engines is None:
            engines = [XDNA2Engine()]
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryXDNA2Platform(MemoryPlatform):
    """XDNA2 platform with memory hierarchy support for tiling.

    Defines the memory hierarchy:
    - L1: 8KB per AIE core (local memory)
    - L3: Shared memory for entire AIE array
    """

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 engines = None,
                 variableBuffer = XDNA2VariableBuffer,
                 constantBuffer = XDNA2ConstantBuffer,
                 structBuffer = XDNA2StructBuffer,
                 transientBuffer = XDNA2TransientBuffer) -> None:
        if engines is None:
            engines = [XDNA2AIECoreEngine()]
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        """Get the target memory level for a tensor in a given node.

        For XDNA2, if the node is marked to run on AIE core engine, return L1 (preferred level).
        Otherwise use the default target memory level (typically L3).
        """
        # Check if node has an engine assignment
        if hasattr(node, '_engine_assignment'):
            engine = node._engine_assignment
            if isinstance(engine, XDNA2AIECoreEngine) and hasattr(engine, 'preferredMemoryLevel'):
                return engine.preferredMemoryLevel

        return self.defaultTargetMemoryLevel.name


class MemoryXDNA2PlatformWrapper(MemoryPlatformWrapper):
    """Wrapper for XDNA2Platform with memory-level support."""

    def __init__(self, platform: XDNA2Platform, memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel):
        assert isinstance(platform, XDNA2Platform), \
            f"Given platform is not an instance of XDNA2Platform. Platform type: {type(platform).__name__}"
        super().__init__(platform, memoryHierarchy, defaultTargetMemoryLevel)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        """Get the target memory level for a tensor in a given node."""
        if hasattr(node, '_engine_assignment'):
            engine = node._engine_assignment
            if isinstance(engine, XDNA2AIECoreEngine) and hasattr(engine, 'preferredMemoryLevel'):
                return engine.preferredMemoryLevel

        return self.defaultTargetMemoryLevel.name
