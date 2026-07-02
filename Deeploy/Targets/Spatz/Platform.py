# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List
import numpy as np

from Deeploy.DeeployTypes import VariableBuffer, TransientBuffer, ConstantBuffer, StructBuffer, \
    NodeMapper, NodeTemplate, TopologyOptimizer, DeploymentEngine, DeploymentPlatform

from Deeploy.Targets.Spatz.Templates import AllocateTemplate as SpatzAllocateTemplate
from Deeploy.Targets.Spatz.Templates import FreeTemplate as SpatzFreeTemplate

from Deeploy.Targets.Spatz.Tiler import SpatzMatMulTilingBindings, SpatzGatherTilingBindings, SpatzTopKTilingBindings, SpatzSoftmaxTilingBindings
from Deeploy.Targets.Generic.Layers import GEMMLayer, SoftmaxLayer, TopKLayer, GatherLayer
from Deeploy.Targets.Generic.Parsers import MatMulParser, SoftmaxParser, TopKParser, GatherParser

MatMulMapper = NodeMapper(MatMulParser(), SpatzMatMulTilingBindings)
SoftmaxMapper = NodeMapper(SoftmaxParser(), SpatzSoftmaxTilingBindings)
TopKMapper = NodeMapper(TopKParser(), SpatzTopKTilingBindings)
GatherMapper = NodeMapper(GatherParser(), SpatzGatherTilingBindings)

SpatzMapping = {
    'MatMul': GEMMLayer([MatMulMapper]),
    'Softmax': SoftmaxLayer([SoftmaxMapper]),
    'TopK': TopKLayer([TopKMapper]),
    'Gather': GatherLayer([GatherMapper]),
}


class SpatzVariableBuffer(VariableBuffer):
    initTemplate = SpatzAllocateTemplate.spatzInitTemplate
    allocTemplate = SpatzAllocateTemplate.spatzGenericAllocate
    deallocTemplate = SpatzFreeTemplate.spatzLocalTemplate

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

class SpatzTransientBuffer(TransientBuffer):
    initTemplate = SpatzAllocateTemplate.spatzInitTemplate
    allocTemplate = SpatzAllocateTemplate.spatzGenericAllocate
    deallocTemplate = SpatzFreeTemplate.spatzLocalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._type,
            "name": self.name,
            "size": self.size,
            "_memoryLevel": memoryLevel
        }


class SpatzConstantBuffer(ConstantBuffer):
    initTemplate = SpatzAllocateTemplate.spatzGlobalInitTemplate
    allocTemplate = NodeTemplate("")
    deallocTemplate = NodeTemplate("")

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class SpatzStructBuffer(StructBuffer):
    initTemplate = SpatzAllocateTemplate.spatzStructInitTemplate
    allocTemplate = SpatzAllocateTemplate.spatzStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


SpatzOptimizer = TopologyOptimizer([
], name = "SpatzOptimizer")

includeList = [
    "snrt.h",
    "DeeploySpatzMath.h",
]


class SpatzEngine(DeploymentEngine):
    def __init__(self, name: str, Mapping = SpatzMapping, initCode = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SpatzPlatform(DeploymentPlatform):

    def __init__( self,
        engines = [SpatzEngine("SpatzVectorProcessor")],
        variableBuffer = SpatzVariableBuffer,
        transientBuffer = SpatzTransientBuffer,
        constantBuffer = SpatzConstantBuffer,
        structBuffer = SpatzStructBuffer,
        includeList: List[str] = includeList
    ):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)

