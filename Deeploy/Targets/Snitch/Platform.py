# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from typing import List, Type

import numpy as np

from Deeploy.AbstractDataTypes import Pointer, PointerClass, VoidType
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicConcatBindings, BasicGatherBindings, BasicLayerNormBindings, \
    BasicMatMulBindings, BasicPad1DBindings, BasicPad2DBindings, BasicReshapeBindings, BasicRQIntegerDivBinding, \
    BasicTransposeBindings
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
    BasicRMSNormBindings, SnitchAddBindings, SnitchGemmBindings, SnitchiNoNormBindings, SnitchiSoftmaxBindings, \
    SnitchRQAddBindings, SnitchRqGemmBindings
from Deeploy.Targets.Snitch.Parsers import HardSwishParser, SnitchDivParser, SnitchGEMMParser, SnitchMulParser, \
    SnitchRMSNormParser, SnitchRQGEMMParser
from Deeploy.Targets.Snitch.Templates import AllocateTemplate, FreeTemplate

# =============================================================================
# Mappers for UNTILED mode (using BasicBindings with BasicTransformer)
# These are used by generateNetwork.py (testRunner_snitch.py)
# =============================================================================
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
ConcatMapper = NodeMapper(ConcatParser(), BasicConcatBindings)

RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])

# These use TiledTransformer but work in both modes (original upstream behavior)
GemmMapper = NodeMapper(SnitchGEMMParser(), SnitchGemmBindings)
RqGemmMapper = NodeMapper(SnitchRQGEMMParser(), SnitchRqGemmBindings)
iSoftmaxMapper = NodeMapper(iSoftmaxParser(), SnitchiSoftmaxBindings)
SoftmaxMapper = NodeMapper(SoftmaxParser(), SnitchiSoftmaxBindings)
iNoNormMapper = NodeMapper(iNoNormParser(), SnitchiNoNormBindings)
iLayerNormMapper = NodeMapper(iLayerNormParser(), BasicLayerNormBindings)
RQAddMapper = NodeMapper(RQAddParser(), SnitchRQAddBindings)
AddMapper = NodeMapper(AddParser(), SnitchAddBindings)

# New operators for microLlama - using BasicBindings for untiled mode
RMSNormMapper = NodeMapper(SnitchRMSNormParser(), BasicRMSNormBindings)
HardSwishMapper = NodeMapper(HardSwishParser(), BasicHardSwishBindings)
MatMulMapper = NodeMapper(MatMulParser(), BasicMatMulBindings)
DivMapper = NodeMapper(SnitchDivParser(), BasicDivBindings)
MulMapper = NodeMapper(SnitchMulParser(), BasicMulBindings)

# =============================================================================
# SnitchMapping - for UNTILED mode (generateNetwork.py)
# Uses BasicBindings for new operators, TiledTransformer bindings for original ops
# =============================================================================
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

# =============================================================================
# Import TilingReadyBindings for TILED mode (testMVP.py)
# These will be used by TilerDeployerWrapper
# =============================================================================
from Deeploy.Targets.Snitch.Tiler import SnitchAddTileReadyBindings, SnitchConcatTilingReadyBindings, \
    SnitchDivTilingReadyBindings, SnitchGatherTilingReadyBindings, SnitchGemmTilingReadyBindings, \
    SnitchHardSwishTilingReadyBindings, SnitchiNoNormTilingReadyBindings, SnitchiSoftmaxTilingReadyBindings, \
    SnitchMatMulTilingReadyBindings, SnitchMulTilingReadyBindings, SnitchReshapeTilingReadyBindings, \
    SnitchRMSNormTilingReadyBindings, SnitchRQAddTilingReadyBindings, SnitchRqGemmTilingReadyBindings, \
    SnitchTransposeTilingReadyBindings

# =============================================================================
# Tiled Mappers - for TILED mode (testMVP.py via TilerDeployerWrapper)
# =============================================================================
TiledGatherMapper = NodeMapper(GatherParser(), SnitchGatherTilingReadyBindings)
TiledUnsqueezeMapper = NodeMapper(UnsqueezeParser(), SnitchReshapeTilingReadyBindings)
TiledReshapeMapper = NodeMapper(ReshapeParser(), SnitchReshapeTilingReadyBindings)
TiledTransposeMapper = NodeMapper(TransposeParser(), SnitchTransposeTilingReadyBindings)
TiledConcatMapper = NodeMapper(ConcatParser(), SnitchConcatTilingReadyBindings)
TiledMatMulMapper = NodeMapper(MatMulParser(), SnitchMatMulTilingReadyBindings)
TiledRMSNormMapper = NodeMapper(SnitchRMSNormParser(), SnitchRMSNormTilingReadyBindings)
TiledHardSwishMapper = NodeMapper(HardSwishParser(), SnitchHardSwishTilingReadyBindings)
TiledDivMapper = NodeMapper(SnitchDivParser(), SnitchDivTilingReadyBindings)
TiledMulMapper = NodeMapper(SnitchMulParser(), SnitchMulTilingReadyBindings)
TiledGemmMapper = NodeMapper(SnitchGEMMParser(), SnitchGemmTilingReadyBindings)
TiledRqGemmMapper = NodeMapper(SnitchRQGEMMParser(), SnitchRqGemmTilingReadyBindings)
TilediSoftmaxMapper = NodeMapper(iSoftmaxParser(), SnitchiSoftmaxTilingReadyBindings)
TiledSoftmaxMapper = NodeMapper(SoftmaxParser(), SnitchiSoftmaxTilingReadyBindings)
TilediNoNormMapper = NodeMapper(iNoNormParser(), SnitchiNoNormTilingReadyBindings)
TiledRQAddMapper = NodeMapper(RQAddParser(), SnitchRQAddTilingReadyBindings)
TiledAddMapper = NodeMapper(AddParser(), SnitchAddTileReadyBindings)

# =============================================================================
# SnitchTiledMapping - for TILED mode (testMVP.py)
# Uses TilingReadyBindings for all operators
# =============================================================================
SnitchTiledMapping = {
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Gather': GatherLayer([TiledGatherMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Unsqueeze': ReshapeLayer([TiledUnsqueezeMapper]),
    'MatMul': MatMulLayer([TiledMatMulMapper]),
    'Gemm': GEMMLayer([TiledGemmMapper]),
    'RQGemm': RQGEMMLayer([TiledRqGemmMapper]),
    'iSoftmax': SoftmaxLayer([TilediSoftmaxMapper]),
    'Softmax': SoftmaxLayer([TiledSoftmaxMapper]),
    'iNoNorm': iNoNormLayer([TilediNoNormMapper]),
    'iLayerNorm': LayerNormLayer([iLayerNormMapper]),
    'RequantizedAdd': AddLayer([TiledRQAddMapper]),
    'Add': AddLayer([TiledAddMapper]),
    'RMSNorm': RMSNormLayer([TiledRMSNormMapper]),
    'HardSwish': HardSwishLayer([TiledHardSwishMapper]),
    'Div': DivLayer([TiledDivMapper]),
    'Mul': MulLayer([TiledMulMapper]),
    'Reshape': ReshapeLayer([TiledReshapeMapper]),
    'Transpose': TransposeLayer([TiledTransposeMapper]),
    'Concat': ConcatLayer([TiledConcatMapper]),
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


class SnitchTiledClusterEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = SnitchTiledMapping, initCode = "", includeList = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class SnitchTiledPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [SnitchTiledClusterEngine("SnitchCluster")],
                 variableBuffer = SnitchVariableBuffer,
                 constantBuffer = SnitchConstantBuffer,
                 structBuffer = SnitchStructBuffer,
                 transientBuffer = SnitchTransientBuffer,
                 includeList: List[str] = _includeList):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
