from typing import List

from Deeploy.DeeployTypes import VariableBuffer, TransientBuffer, ConstantBuffer, StructBuffer, \
    NodeMapper, NodeTemplate, TopologyOptimizer, DeploymentEngine, DeploymentPlatform

# from Deeploy.Targets.Spatz.Bindings import SpatzAddBindings # <- TODO create this
from Deeploy.Targets.Generic.Bindings import BasicAddBindings
from Deeploy.Targets.Generic.Layers import AddLayer
from Deeploy.Targets.Generic.Parsers import AddParser

from Deeploy.Targets.Generic.Templates import AllocateTemplate as GenericAllocateTemplate
from Deeploy.Targets.Spatz.Templates import AllocateTemplate as SpatzAllocateTemplate
from Deeploy.Targets.Spatz.Templates import FreeTemplate as SpatzFreeTemplate

SpatzAddMapper = NodeMapper(AddParser(), BasicAddBindings)

SpatzMapping = {
    'Add': AddLayer([SpatzAddMapper]),
    # sparse attention : ...
}


class SpatzaVariableBuffer(VariableBuffer):
    initTemplate = GenericAllocateTemplate.referenceInitTemplate
    allocTemplate = SpatzAllocateTemplate.referenceAllocateTemplate
    deallocTemplate = SpatzFreeTemplate.spatzLocalTemplate


class SpatzTransientBuffer(TransientBuffer):
    initTemplate = GenericAllocateTemplate.referenceInitTemplate
    allocTemplate = SpatzAllocateTemplate.referenceAllocateTemplate
    deallocTemplate = SpatzFreeTemplate.spatzLocalTemplate


class SpatzConstantBuffer(ConstantBuffer):
    initTemplate = GenericAllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = GenericAllocateTemplate.referenceAllocateTemplate
    deallocTemplate = NodeTemplate("") # const not deallocated


class SpatzStructBuffer(StructBuffer):
    initTemplate = GenericAllocateTemplate.referenceStructInitTemplate
    allocTemplate = GenericAllocateTemplate.referenceAllocateTemplate
    deallocTemplate = NodeTemplate("") # struct not deallocated ?


SpatzOptimizer = TopologyOptimizer([
    # TODO add something ?
], name = "SpatzOptimizer")

includeList = [
    "DeeploySpatzMath.h",
]


class SpatzEngine(DeploymentEngine):
    def __init__(self, name: str, Mapping = SpatzMapping, initCode = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SpatzPlatform(DeploymentPlatform):

    def __init__( self,
        engines = [SpatzEngine("SpatzVectorProcessor")],
        variableBuffer = SpatzaVariableBuffer,
        transientBuffer = SpatzTransientBuffer,
        constantBuffer = SpatzConstantBuffer,
        structBuffer = SpatzStructBuffer,
        includeList: List[str] = includeList
    ):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
