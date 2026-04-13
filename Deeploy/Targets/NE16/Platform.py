# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import Optional

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RequantizedGemmToPwPass
from Deeploy.DeeployTypes import TopologyOptimizer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.Targets.GAP9.Platform import GAP9ClusterEngine, GAP9ConstantBuffer, GAP9Platform, GAP9StructBuffer, \
    GAP9TransientBuffer, GAP9VariableBuffer, MemoryGAP9Platform, MemoryGAP9PlatformWrapper
from Deeploy.Targets.NE16.Engine import NE16Engine
from Deeploy.Targets.PULPOpen.Platform import PULPOptimizer

NE16Optimizer = TopologyOptimizer([
    *PULPOptimizer.passes,
    RequantizedGemmToPwPass(),
], name = "NE16Optimizer")


class NE16Platform(GAP9Platform):

    def __init__(self,
                 engines = None,
                 variableBuffer = GAP9VariableBuffer,
                 constantBuffer = GAP9ConstantBuffer,
                 structBuffer = GAP9StructBuffer,
                 transientBuffer = GAP9TransientBuffer) -> None:
        if engines is None:
            engines = [NE16Engine("NE16"), GAP9ClusterEngine("GAP9Cluster")]
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryNE16Platform(MemoryGAP9Platform):

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 weightMemoryLevel: Optional[MemoryLevel] = None,
                 engines = None,
                 variableBuffer = GAP9VariableBuffer,
                 constantBuffer = GAP9ConstantBuffer,
                 structBuffer = GAP9StructBuffer,
                 transientBuffer = GAP9TransientBuffer) -> None:
        if engines is None:
            engines = [NE16Engine("NE16"), GAP9ClusterEngine("GAP9Cluster")]
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)
        self.weightMemoryLevel = weightMemoryLevel


class MemoryNE16PlatformWrapper(MemoryGAP9PlatformWrapper):

    def __init__(self,
                 platform: NE16Platform,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 weightMemoryLevel: Optional[MemoryLevel] = None):
        assert isinstance(platform, NE16Platform), \
            f"Given platform is not an instance of NE16Platform. Platform type: {type(platform).__name__}"
        super().__init__(platform, memoryHierarchy, defaultTargetMemoryLevel)
        self.weightMemoryLevel = weightMemoryLevel
