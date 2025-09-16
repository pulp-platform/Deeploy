# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Chimera.Templates import AllocateTemplate as ChimeraAllocateTemplate
from Deeploy.Targets.Generic.Bindings import BasicAddBindings
from Deeploy.Targets.Generic.Layers import AddLayer
from Deeploy.Targets.Generic.Parsers import AddParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate

AddMapper = NodeMapper(AddParser(), BasicAddBindings)

ChimeraMapping = {
    'Add': AddLayer([AddMapper]),
}


class ChimeraVariableBuffer(VariableBuffer):

    initTemplate = BasicAllocateTemplate.referenceInitTemplate
    allocTemplate = ChimeraAllocateTemplate.memoryIslandAllocateTemplate
    deallocTemplate = ChimeraAllocateTemplate.memoryIslandFreeTemplate


class ChimeraTransientBuffer(TransientBuffer):

    initTemplate = BasicAllocateTemplate.referenceInitTemplate
    allocTemplate = ChimeraAllocateTemplate.memoryIslandAllocateTemplate
    deallocTemplate = ChimeraAllocateTemplate.memoryIslandFreeTemplate


class ChimeraConstantBuffer(ConstantBuffer):

    initTemplate = BasicAllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceGlobalInitTemplate
    deallocTemplate = NodeTemplate("")


class ChimeraStructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


ChimeraOptimizer = TopologyOptimizer([
    # JUNGVI: Nothing for now
])

_includeList = [
    "uart.h",
    "DeeployChimeraMath.h",
]


class ChimeraHostEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = ChimeraMapping, initCode = "", includeList = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class ChimeraPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [ChimeraHostEngine("ChimeraHost")],
                 variableBuffer = ChimeraVariableBuffer,
                 constantBuffer = ChimeraConstantBuffer,
                 structBuffer = ChimeraStructBuffer,
                 transientBuffer = ChimeraTransientBuffer,
                 includeList: List[str] = _includeList):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
