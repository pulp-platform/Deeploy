# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Layers import AddLayer
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Generic.Parsers import AddParser
from Deeploy.Targets.XDNA2.Bindings import XDNA2AddBindings

XDNA2AddMapper = NodeMapper(AddParser(), XDNA2AddBindings)

XDNA2Mapping = {
    'Add': AddLayer([XDNA2AddMapper]),
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

    def __init__(self, name: str = "XDNA2", Mapping = XDNA2Mapping, initCode: str = "",
                 includeList = None) -> None:
        if includeList is None:
            includeList = []
        super().__init__(name, Mapping, initCode, includeList)


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
