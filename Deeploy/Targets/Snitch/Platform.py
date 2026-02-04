# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Type

import numpy as np

from Deeploy.AbstractDataTypes import Pointer
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.AbstractDataTypes import VoidType
from Deeploy.DeeployTypes import ConstantBuffer
from Deeploy.DeeployTypes import DeploymentEngine
from Deeploy.DeeployTypes import DeploymentPlatform
from Deeploy.DeeployTypes import NodeMapper
from Deeploy.DeeployTypes import NodeTemplate
from Deeploy.DeeployTypes import StructBuffer
from Deeploy.DeeployTypes import TopologyOptimizer
from Deeploy.DeeployTypes import TransientBuffer
from Deeploy.DeeployTypes import VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicConcatBindings
from Deeploy.Targets.Generic.Bindings import BasicGatherBindings
from Deeploy.Targets.Generic.Bindings import BasicLayerNormBindings
from Deeploy.Targets.Generic.Bindings import BasicMatMulBindings
from Deeploy.Targets.Generic.Bindings import BasicPad1DBindings
from Deeploy.Targets.Generic.Bindings import BasicPad2DBindings
from Deeploy.Targets.Generic.Bindings import BasicReshapeBindings
from Deeploy.Targets.Generic.Bindings import BasicRQIntegerDivBinding
from Deeploy.Targets.Generic.Layers import AddLayer
from Deeploy.Targets.Generic.Layers import ConcatLayer
from Deeploy.Targets.Generic.Layers import DivLayer
from Deeploy.Targets.Generic.Layers import GatherLayer
from Deeploy.Targets.Generic.Layers import GEMMLayer
from Deeploy.Targets.Generic.Layers import HardSwishLayer
from Deeploy.Targets.Generic.Layers import iNoNormLayer
from Deeploy.Targets.Generic.Layers import LayerNormLayer
from Deeploy.Targets.Generic.Layers import MatMulLayer
from Deeploy.Targets.Generic.Layers import MulLayer
from Deeploy.Targets.Generic.Layers import PadLayer
from Deeploy.Targets.Generic.Layers import ReshapeLayer
from Deeploy.Targets.Generic.Layers import RMSNormLayer
from Deeploy.Targets.Generic.Layers import RQGEMMLayer
from Deeploy.Targets.Generic.Layers import RQIntegerDivLayer
from Deeploy.Targets.Generic.Layers import SoftmaxLayer
from Deeploy.Targets.Generic.Layers import TransposeLayer
from Deeploy.Targets.Generic.Parsers import AddParser
from Deeploy.Targets.Generic.Parsers import ConcatParser
from Deeploy.Targets.Generic.Parsers import GatherParser
from Deeploy.Targets.Generic.Parsers import iLayerNormParser
from Deeploy.Targets.Generic.Parsers import iNoNormParser
from Deeploy.Targets.Generic.Parsers import iSoftmaxParser
from Deeploy.Targets.Generic.Parsers import MatMulParser
from Deeploy.Targets.Generic.Parsers import Pad1DParser
from Deeploy.Targets.Generic.Parsers import Pad2DParser
from Deeploy.Targets.Generic.Parsers import ReshapeParser
from Deeploy.Targets.Generic.Parsers import RQAddParser
from Deeploy.Targets.Generic.Parsers import RQIntegerDivParser
from Deeploy.Targets.Generic.Parsers import SoftmaxParser
from Deeploy.Targets.Generic.Parsers import TransposeParser
from Deeploy.Targets.Generic.Parsers import UnsqueezeParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import AddRequantMergePass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import GEMMRequantMergePass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import iGELURequantMergePass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import iHardswishRequantMergePass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import IntegerDivRequantMergePass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import MergeConstAddAndRequantPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import MergeTrueIntegerDivRequantShiftPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import RQSSplitPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import SkipEmptyConcatPass
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import SkipUnityRequantPass
from Deeploy.Targets.PULPOpen.Platform import RQAddMapper
from Deeploy.Targets.Snitch.Bindings import BasicDivBindings
from Deeploy.Targets.Snitch.Bindings import BasicHardSwishBindings
from Deeploy.Targets.Snitch.Bindings import BasicMulBindings
from Deeploy.Targets.Snitch.Bindings import BasicRMSNormBindings
from Deeploy.Targets.Snitch.Bindings import BasicSnitchTransposeBindings
from Deeploy.Targets.Snitch.Bindings import SnitchAddBindings
from Deeploy.Targets.Snitch.Bindings import SnitchGemmBindings
from Deeploy.Targets.Snitch.Bindings import SnitchiNoNormBindings
from Deeploy.Targets.Snitch.Bindings import SnitchiSoftmaxBindings
from Deeploy.Targets.Snitch.Bindings import SnitchRQAddBindings
from Deeploy.Targets.Snitch.Bindings import SnitchRqGemmBindings
from Deeploy.Targets.Snitch.Parsers import HardSwishParser
from Deeploy.Targets.Snitch.Parsers import SnitchDivParser
from Deeploy.Targets.Snitch.Parsers import SnitchGEMMParser
from Deeploy.Targets.Snitch.Parsers import SnitchMulParser
from Deeploy.Targets.Snitch.Parsers import SnitchRMSNormParser
from Deeploy.Targets.Snitch.Parsers import SnitchRQGEMMParser
from Deeploy.Targets.Snitch.Templates import AllocateTemplate
from Deeploy.Targets.Snitch.Templates import FreeTemplate
from Deeploy.Targets.Snitch.Tiler import SnitchAddTileReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchConcatTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchDivTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchGatherTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchGemmTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchHardSwishTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchiNoNormTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchiSoftmaxTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchMatMulTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchMulTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchReshapeTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchRMSNormTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchRQAddTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchRqGemmTilingReadyBindings
from Deeploy.Targets.Snitch.Tiler import SnitchTransposeTilingReadyBindings

# Mappers without tiling-ready equivalents
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
TransposeMapper = NodeMapper(TransposeParser(), BasicSnitchTransposeBindings)
ConcatMapper = NodeMapper(ConcatParser(), BasicConcatBindings)

RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
iLayerNormMapper = NodeMapper(iLayerNormParser(), BasicLayerNormBindings)

# All other mappers use TilingReadyBindings (works for both tiled and untiled)
GatherMapper = NodeMapper(GatherParser(), SnitchGatherTilingReadyBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), SnitchReshapeTilingReadyBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), SnitchReshapeTilingReadyBindings)
TransposeMapper = NodeMapper(TransposeParser(), SnitchTransposeTilingReadyBindings)
ConcatMapper = NodeMapper(ConcatParser(), SnitchConcatTilingReadyBindings)
MatMulMapper = NodeMapper(MatMulParser(), SnitchMatMulTilingReadyBindings)
GemmMapper = NodeMapper(SnitchGEMMParser(), SnitchGemmTilingReadyBindings)
RqGemmMapper = NodeMapper(SnitchRQGEMMParser(), SnitchRqGemmTilingReadyBindings)
iSoftmaxMapper = NodeMapper(iSoftmaxParser(), SnitchiSoftmaxTilingReadyBindings)
SoftmaxMapper = NodeMapper(SoftmaxParser(), SnitchiSoftmaxTilingReadyBindings)
iNoNormMapper = NodeMapper(iNoNormParser(), SnitchiNoNormTilingReadyBindings)
RQAddMapper = NodeMapper(RQAddParser(), SnitchRQAddTilingReadyBindings)
AddMapper = NodeMapper(AddParser(), SnitchAddTileReadyBindings)
RMSNormMapper = NodeMapper(SnitchRMSNormParser(), SnitchRMSNormTilingReadyBindings)
HardSwishMapper = NodeMapper(HardSwishParser(), SnitchHardSwishTilingReadyBindings)
DivMapper = NodeMapper(SnitchDivParser(), SnitchDivTilingReadyBindings)
MulMapper = NodeMapper(SnitchMulParser(), SnitchMulTilingReadyBindings)

SnitchMapping = {
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'Gemm': GEMMLayer([GemmMapper]),
    'RQGemm': RQGEMMLayer([RqGemmMapper]),
    'iSoftmax': SoftmaxLayer([iSoftmaxMapper]),
    'Softmax': SoftmaxLayer([SoftmaxMapper]),
    'iNoNorm': iNoNormLayer([iNoNormMapper]),
    'iLayerNorm': LayerNormLayer([iLayerNormMapper]),
    'RequantizedAdd': AddLayer([RQAddMapper]),
    'Add': AddLayer([AddMapper]),
    'RMSNorm': RMSNormLayer([RMSNormMapper]),
    'HardSwish': HardSwishLayer([HardSwishMapper]),
    'Div': DivLayer([DivMapper]),
    'Mul': MulLayer([MulMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Concat': ConcatLayer([ConcatMapper]),
}


class SnitchVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.snitchL2InitTemplate
    allocTemplate = AllocateTemplate.snitchGenericAllocate
    deallocTemplate = FreeTemplate.snitchGenericFree

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


class SnitchTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.snitchL2InitTemplate
    allocTemplate = AllocateTemplate.snitchGenericAllocate
    deallocTemplate = FreeTemplate.snitchGenericFree

    # allocTemplate = AllocateTemplate.snitchL2AllocateTemplate
    # deallocTemplate = FreeTemplate.snitchL2GlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}


class SnitchConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.snitchGenericGlobalInitTemplate
    allocTemplate = AllocateTemplate.snitchL2GlobalAllocateTemplate
    deallocTemplate = FreeTemplate.snitchL2GlobalTemplate

    def __init__(self, name: str = '', shape = [1], values = [0]):
        super().__init__(name, shape, values)
        # Initialize _type with a default value to prevent AttributeError
        # The actual type will be set later via annotateType
        self._type: Type[Pointer] = PointerClass(VoidType)

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class SnitchStructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


SnitchOptimizer = TopologyOptimizer([
    SkipEmptyConcatPass(),
    SkipUnityRequantPass(previous_op_regex = "Concat", num_inputs = 2),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    RQSSplitPass(),
    MergeTrueIntegerDivRequantShiftPass(),
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    iHardswishRequantMergePass(),
    MergeConstAddAndRequantPass(),
    AddRequantMergePass(),
    GEMMRequantMergePass(),
],
                                    name = "SnitchOptimizer")

_includeList = [
    "snrt.h",
    "DeeploySnitchMath.h",
]


class SnitchClusterEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SnitchMapping, initCode = "", includeList = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SnitchPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [SnitchClusterEngine("SnitchCluster")],
                 variableBuffer = SnitchVariableBuffer,
                 constantBuffer = SnitchConstantBuffer,
                 structBuffer = SnitchStructBuffer,
                 transientBuffer = SnitchTransientBuffer,
                 includeList: List[str] = _includeList):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
