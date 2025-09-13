# Copyright (C) 2025, ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.CortexM.Bindings import CMSISCLCABinding, CMSISConv1DBindings, CMSISConv2DBinding, \
    CMSISDWConv1DBindings, CMSISDWConv2DBinding, CMSISGEMMBindings, CMSISLinearAttentionBinding, \
    CMSISMaxPool2DBinding
from Deeploy.Targets.CortexM.Layers import CMSISRQSConvLayer, CMSISRQSGEMMLayer
from Deeploy.Targets.CortexM.Parsers import CMSISCLCAParser, CMSISConv1DParser, CMSISConv2DParser, \
    CMSISDWConv1DParser, CMSISDWConv2DParser, CMSISGEMMParser, CMSISLinearAttentionParser, CMSISMaxPool2DParser
from Deeploy.Targets.CortexM.TopologyOptimizationPasses.Passes import ConvRequantMergePass, GEMMRequantMergePass, \
    LinearAttentionAlignmentPass, MatMulRequantMergePass, MHSAAlignmentPass
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicDebugPrintBindings, BasicDivBindings, \
    BasicGatherBindings, BasicGELUBindings, BasicLayerNormBindings, BasicMatMulBindings, BasicMulBindings, \
    BasicPad1DBindings, BasicPad2DBindings, BasicReduceMeanBindings, BasicReduceSumBindings, BasicReshapeBindings, \
    BasicRQIntegerDivBinding, BasicRQSBindings, BasicRQSGELUBinding, BasicSliceBindings, BasicSoftmaxBindings, \
    BasicTransposeBindings, DummyBinding
from Deeploy.Targets.Generic.Layers import AddLayer, CLCALayer, DebugPrintLayer, DivLayer, GatherLayer, GELULayer, \
    LayerNormLayer, LinearAttentionLayer, MatMulLayer, MaxPoolLayer, MulLayer, PadLayer, ReduceMeanLayer, \
    ReduceSumLayer, RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, SliceLayer, SoftmaxLayer, \
    TransposeLayer
from Deeploy.Targets.Generic.Parsers import AddParser, DebugParser, DummyParser, FlattenParser, GatherParser, \
    GELUParser, IntegerDivParser, MatMulParser, MulParser, Pad1DParser, Pad2DParser, ReduceMeanParser, \
    ReduceSumParser, RequantShiftParser, ReshapeParser, RQIntegerDivParser, RQSiGELUParser, SliceParser, \
    TransposeParser, UnsqueezeParser, iLayerNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import IntegerDivRequantMergePass, \
    MergeConstAddAndRequantPass, iGELURequantMergePass

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
CLCA_int8_Mapper = NodeMapper(CMSISCLCAParser(), [CMSISCLCABinding])
Conv1D_Mapper = NodeMapper(CMSISConv1DParser(), CMSISConv1DBindings)
Conv2D_int8_Mapper = NodeMapper(CMSISConv2DParser(), [CMSISConv2DBinding])
DebugPrint_Mapper = NodeMapper(DebugParser(), BasicDebugPrintBindings)
DWConv1D_Mapper = NodeMapper(CMSISDWConv1DParser(), CMSISDWConv1DBindings)
DWConv2D_int8_Mapper = NodeMapper(CMSISDWConv2DParser(), [CMSISDWConv2DBinding])
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELU_int8_Mapper = NodeMapper(GELUParser(), BasicGELUBindings)
GEMMMapper = NodeMapper(CMSISGEMMParser(), CMSISGEMMBindings)
iLayerNorm_int8_Mapper = NodeMapper(iLayerNormParser(), BasicLayerNormBindings)
IntegerDivMapper = NodeMapper(IntegerDivParser(), BasicDivBindings)
LinearAttention_int16_Mapper = NodeMapper(CMSISLinearAttentionParser(), [CMSISLinearAttentionBinding])
MatMulMapper = NodeMapper(MatMulParser(), BasicMatMulBindings)
MaxPool2DMapper = NodeMapper(CMSISMaxPool2DParser(), [CMSISMaxPool2DBinding])
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
ReduceSumMapper = NodeMapper(ReduceSumParser(), BasicReduceSumBindings)
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), BasicSoftmaxBindings)
TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

SliceMapper = NodeMapper(SliceParser(), BasicSliceBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

CMSISMapping = {
    'Add': AddLayer([AddMapper]),
    'CLCA': CLCALayer([CLCA_int8_Mapper]),
    'DebugPrint': DebugPrintLayer([DebugPrint_Mapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'iGELU': GELULayer([GELU_int8_Mapper]),
    'iLayerNorm': LayerNormLayer([iLayerNorm_int8_Mapper]),
    'IntegerDiv': DivLayer([IntegerDivMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': SoftmaxLayer([Softmax_int8_Mapper]),
    'LinearAttention': LinearAttentionLayer([LinearAttention_int16_Mapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'MaxPool': MaxPoolLayer([MaxPool2DMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'ReduceSum': ReduceSumLayer([ReduceSumMapper]),
    'RequantizedConv': CMSISRQSConvLayer([Conv2D_int8_Mapper, DWConv2D_int8_Mapper, Conv1D_Mapper, DWConv1D_Mapper]),
    'RequantizedGemm': CMSISRQSGEMMLayer([GEMMMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_int8_Mapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Slice': SliceLayer([SliceMapper])
}


class CMSISVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class CMSISTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class CMSISConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = AllocateTemplate.referenceGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.referenceGlobalTemplate


class CMSISStructBuffer(StructBuffer):

    initTemplate = AllocateTemplate.referenceStructInitTemplate
    allocTemplate = AllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


# ExtractPaddingFromConvPass(),ExtractPaddingFromPoolPass(),
CMSISOptimizer = TopologyOptimizer([
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    LinearAttentionAlignmentPass(),
    MHSAAlignmentPass(),
    MergeConstAddAndRequantPass(),
    ConvRequantMergePass(),
    GEMMRequantMergePass(),
    MatMulRequantMergePass(),
    # DebugPass("Conv", position='before'),
    # DebugPass("Pad", position='after'),
])

includeList = ["arm_nnfunctions.h", "DeeployMath.h"]


class CMSISEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = CMSISMapping, initCode: str = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class CMSISPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [CMSISEngine("cmsis")],
                 variableBuffer = CMSISVariableBuffer,
                 constantBuffer = CMSISConstantBuffer,
                 structBuffer = CMSISStructBuffer,
                 transientBuffer = CMSISTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
