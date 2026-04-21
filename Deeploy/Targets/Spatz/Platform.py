from typing import List
import numpy as np

from Deeploy.DeeployTypes import VariableBuffer, TransientBuffer, ConstantBuffer, StructBuffer, \
    NodeMapper, NodeTemplate, TopologyOptimizer, DeploymentEngine, DeploymentPlatform

from Deeploy.Targets.Generic.Templates import AllocateTemplate as GenericAllocateTemplate
from Deeploy.Targets.Spatz.Templates import AllocateTemplate as SpatzAllocateTemplate
from Deeploy.Targets.Spatz.Templates import FreeTemplate as SpatzFreeTemplate
from Deeploy.Targets.Snitch.Templates import AllocateTemplate as SnitchAllocateTemplate, FreeTemplate as SnitchFreeTemplate

from Deeploy.Targets.Spatz.Bindings import SpatzGatherBindings, SpatzMatMulBindings
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicMatMulBindings, BasicSoftmaxBindings, BasicTopKBindings
from Deeploy.Targets.Spatz.Tiler import SpatzMatMulTilingReadyBindings
from Deeploy.Targets.Generic.Layers import AddLayer, GEMMLayer, SoftmaxLayer, TopKLayer, GatherLayer
from Deeploy.Targets.Generic.Parsers import AddParser, MatMulParser, SoftmaxParser, TopKParser, GatherParser

# # print(SpatzMatMulBindings)
# # for binding in SpatzMatMulBindings:
# #     print(binding.template.tileConstraint)
# 
# print(SpatzMatMulTilingReadyBindings)
# for binding in SpatzMatMulTilingReadyBindings:
#     print(binding.template.tileConstraint)
# 
# print(SpatzMatMulTilingReadyBindings[0].template.tileConstraint)
# print(SpatzMatMulTilingReadyBindings[1].template.tileConstraint)

SpatzAddMapper = NodeMapper(AddParser(), BasicAddBindings)
MatMulMapper = NodeMapper(MatMulParser(), SpatzMatMulTilingReadyBindings)
SoftmaxMapper = NodeMapper(SoftmaxParser(), BasicSoftmaxBindings)
TopKMapper = NodeMapper(TopKParser(), BasicTopKBindings)
GatherMapper = NodeMapper(GatherParser(), SpatzGatherBindings)

SpatzMapping = {
    'Add': AddLayer([SpatzAddMapper]),
    'MatMul': GEMMLayer([MatMulMapper]),
    'Softmax': SoftmaxLayer([SoftmaxMapper]),
    'TopK': TopKLayer([TopKMapper]),
    'Gather': GatherLayer([GatherMapper]),
    # sparse attention : ...
}


class SpatzaVariableBuffer(VariableBuffer):
    initTemplate = GenericAllocateTemplate.referenceInitTemplate
    allocTemplate = SpatzAllocateTemplate.referenceAllocateTemplate
    # allocTemplate = SnitchAllocateTemplate.snitchGenericAllocate
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
    initTemplate = GenericAllocateTemplate.referenceInitTemplate
    allocTemplate = SpatzAllocateTemplate.referenceAllocateTemplate
    deallocTemplate = SpatzFreeTemplate.spatzLocalTemplate
#     def _bufferRepresentation(self):
# 
#         if hasattr(self, "_memoryLevel"):
#             memoryLevel = self._memoryLevel
#         else:
#             memoryLevel = None
# 
#         return {
#             "type": self._type,
#             "name": self.name,
#             "size": self.size,
#             "_memoryLevel": memoryLevel
#         }


class SpatzConstantBuffer(ConstantBuffer):
    initTemplate = SnitchAllocateTemplate.snitchGenericGlobalInitTemplate
    allocTemplate = NodeTemplate("")
    deallocTemplate = NodeTemplate("") # const not deallocated

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class SpatzStructBuffer(StructBuffer):
    initTemplate = GenericAllocateTemplate.referenceStructInitTemplate
    allocTemplate = GenericAllocateTemplate.referenceStructAllocateTemplate
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

