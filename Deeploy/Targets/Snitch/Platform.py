# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Type

import numpy as np

from Deeploy.AbstractDataTypes import Pointer, PointerClass, VoidType
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicConcatBindings, BasicGatherBindings, BasicLayerNormBindings, \
    BasicMatMulBindings, BasicPad1DBindings, BasicPad2DBindings, BasicReshapeBindings, BasicRQIntegerDivBinding
from Deeploy.Targets.Generic.Layers import AddLayer, ConcatLayer, DivLayer, GatherLayer, GEMMLayer, HardSwishLayer, \
    LayerNormLayer, MatMulLayer, MulLayer, PadLayer, ReshapeLayer, RMSNormLayer, RQGEMMLayer, RQIntegerDivLayer, \
    SoftmaxLayer, TransposeLayer, iNoNormLayer
from Deeploy.Targets.Generic.Parsers import AddParser, ConcatParser, GatherParser, MatMulParser, Pad1DParser, Pad2DParser, \
    ReshapeParser, RQAddParser, RQIntegerDivParser, SoftmaxParser, TransposeParser, UnsqueezeParser, iLayerNormParser, \
    iNoNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import AddRequantMergePass, GEMMRequantMergePass, \
    IntegerDivRequantMergePass, MergeConstAddAndRequantPass, MergeTrueIntegerDivRequantShiftPass, RQSSplitPass, \
    SkipEmptyConcatPass, SkipUnityRequantPass, iGELURequantMergePass, iHardswishRequantMergePass
from Deeploy.Targets.PULPOpen.Platform import RQAddMapper
from Deeploy.Targets.Snitch.Bindings import BasicDivBindings, BasicHardSwishBindings, BasicMulBindings, \
    BasicRMSNormBindings, BasicSnitchTransposeBindings, SnitchAddBindings, SnitchGemmBindings, \
    SnitchiNoNormBindings, SnitchiSoftmaxBindings, SnitchRQAddBindings, SnitchRqGemmBindings
from Deeploy.Targets.Snitch.Parsers import HardSwishParser, SnitchDivParser, SnitchGEMMParser, SnitchMulParser, \
    SnitchRMSNormParser, SnitchRQGEMMParser
from Deeploy.Targets.Snitch.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Snitch.Tiler import SnitchAddTileReadyBindings, SnitchConcatTilingReadyBindings, \
    SnitchDivTilingReadyBindings, SnitchGatherTilingReadyBindings, SnitchGemmTilingReadyBindings, \
    SnitchHardSwishTilingReadyBindings, SnitchiNoNormTilingReadyBindings, SnitchiSoftmaxTilingReadyBindings, \
    SnitchMatMulTilingReadyBindings, SnitchMulTilingReadyBindings, SnitchReshapeTilingReadyBindings, \
    SnitchRMSNormTilingReadyBindings, SnitchRQAddTilingReadyBindings, SnitchRqGemmTilingReadyBindings, \
    SnitchTransposeTilingReadyBindings

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
