# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NetworkContext, NodeTemplate, \
    StructBuffer, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform, MemoryPlatformWrapper
from Deeploy.Targets.GAP9.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Targets.PULPOpen.Platform import PULPMapping


class GAP9VariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.gap9L2InitTemplate
    # allocTemplate = AllocateTemplate.gap9L2AllocateTemplate
    # deallocTemplate = FreeTemplate.gap9L2LocalTemplate

    allocTemplate = AllocateTemplate.gap9GenericAllocate
    deallocTemplate = FreeTemplate.gap9GenericFree

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class GAP9TransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.gap9L2InitTemplate
    allocTemplate = AllocateTemplate.gap9GenericAllocate
    deallocTemplate = FreeTemplate.gap9GenericFree

    # allocTemplate = AllocateTemplate.gap9L2AllocateTemplate
    # deallocTemplate = FreeTemplate.gap9L2GlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}


class GAP9ConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.gap9GenericGlobalInitTemplate
    allocTemplate = AllocateTemplate.gap9L2GlobalAllocateTemplate
    deallocTemplate = FreeTemplate.gap9L2GlobalTemplate

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class GAP9StructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


# SCHEREMO: stdint is included before pulp_nn_kernels.h because it is supposed to be included in there, but isn't...
_includeList = [
    "pmsis.h",
    "DeeployGAP9Math.h",
    "pulp_nn_kernels.h"
]


class GAP9ClusterEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = PULPMapping, initCode = "", includeList = _includeList, n_cores: int = 8) -> None:
        super().__init__(name, Mapping, initCode, includeList)
        self.n_cores = n_cores


class GAP9Platform(DeploymentPlatform):

    def __init__(self,
                 engines = [GAP9ClusterEngine("GAP9Cluster")],
                 variableBuffer = GAP9VariableBuffer,
                 constantBuffer = GAP9ConstantBuffer,
                 structBuffer = GAP9StructBuffer,
                 transientBuffer = GAP9TransientBuffer) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryGAP9Platform(MemoryPlatform):

    untiledOps = ["add"]

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 engines = [GAP9ClusterEngine("GAP9Cluster")],
                 variableBuffer = GAP9VariableBuffer,
                 constantBuffer = GAP9ConstantBuffer,
                 structBuffer = GAP9StructBuffer,
                 transientBuffer = GAP9TransientBuffer) -> None:
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if node.op in self.untiledOps:
            return ctxt.lookup(tensorName)._memoryLevel
        return super().getTargetMemoryLevel(node, tensorName, ctxt)


class MemoryGAP9PlatformWrapper(MemoryPlatformWrapper):

    untiledOps = ["add"]

    def __init__(self, platform: GAP9Platform, memoryHierarchy: MemoryHierarchy, defaultTargetMemoryLevel: MemoryLevel):
        assert isinstance(platform, GAP9Platform), \
        f"Given platform is not an instance of GAP9Platform. Platform type: {type(platform).__name__}"
        super().__init__(platform, memoryHierarchy, defaultTargetMemoryLevel)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if node.op in self.untiledOps:
            return ctxt.lookup(tensorName)._memoryLevel
        return super().getTargetMemoryLevel(node, tensorName, ctxt)
