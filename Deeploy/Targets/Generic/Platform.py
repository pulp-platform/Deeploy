# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RemoveEmptyConvBiasPass, RemoveOnlySingletonReduceMeanPass
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicBatchNormBindings, BasicConcatBindings, \
    BasicConv1DBindings, BasicConv2DBindings, BasicConvTransposeBindings, BasicDebugPrintBindings, \
    BasicDequantBindings, BasicDivBindings, BasicDWConv1DBinding, BasicDWConv2DBindings, BasicGatherBindings, \
    BasicGELUBindings, BasicGEMMBindings, BasicITAPartialSoftmaxBinding, BasicITASoftmaxBinding, \
    BasicLayerNormBindings, BasicMatMulBindings, BasicMaxPool1DBindings, BasicMaxPool2DBindings, BasicMulBindings, \
    BasicPad1DBindings, BasicPad2DBindings, BasicPowBindings, BasicQuantBindings, BasicReduceMeanBindings, \
    BasicReduceSumBindings, BasicReluBinding, BasicReshapeBindings, BasicRQIntegerDivBinding, BasicRQSBindings, \
    BasicRQSGELUBinding, BasicSliceBindings, BasicSoftmaxBindings, BasicSqrtBindings, BasicTransposeBindings, \
    DummyBinding
from Deeploy.Targets.Generic.Layers import AddLayer, BatchNormalizationLayer, ConcatLayer, ConvLayer, \
    ConvTransposeLayer, DebugPrintLayer, DequantLayer, DivLayer, GatherLayer, GELULayer, GEMMLayer, ITAMaxLayer, \
    LayerNormLayer, MatMulLayer, MaxPoolLayer, MulLayer, PadLayer, PowLayer, QuantLayer, ReduceMeanLayer, \
    ReduceSumLayer, ReluLayer, RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, SliceLayer, \
    SoftmaxLayer, SqrtLayer, TransposeLayer
from Deeploy.Targets.Generic.Parsers import AddParser, BatchNormParser, ConcatParser, ConvTranspose1DParser, \
    DebugParser, DequantParser, DivParser, DummyParser, FlattenParser, GatherParser, GELUParser, GenericConv1DParser, \
    GenericConv2DParser, GenericDWConv1DParser, GenericDWConv2DParser, GenericGEMMParser, GenericMaxPool2DParser, \
    IntegerDivParser, ITAMaxParser, ITAPartialMaxParser, LayerNormParser, MatMulParser, MaxPool1DParser, MulParser, \
    Pad1DParser, Pad2DParser, PowParser, QuantParser, ReduceMeanParser, ReduceSumParser, ReluParser, \
    RequantShiftParser, ReshapeParser, RQIntegerDivParser, RQSiGELUParser, SliceParser, SoftmaxParser, SqrtParser, \
    TransposeParser, UnsqueezeParser, iLayerNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import DequantPatternPass, ExtractPaddingFromConvPass, \
    ExtractPaddingFromPoolPass, MatMulAddMergePass, MergeConstAddAndRequantPass, QuantPatternPass, \
    iGELURequantMergePass

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
Conv1DMapper = NodeMapper(GenericConv1DParser(), BasicConv1DBindings)
Conv2DMapper = NodeMapper(GenericConv2DParser(), BasicConv2DBindings)
ConcatMapper = NodeMapper(ConcatParser(), BasicConcatBindings)
DebugMapper = NodeMapper(DebugParser(), BasicDebugPrintBindings)
DWConv1DMapper = NodeMapper(GenericDWConv1DParser(), [BasicDWConv1DBinding])
DWConv2DMapper = NodeMapper(GenericDWConv2DParser(), BasicDWConv2DBindings)
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELUMapper = NodeMapper(GELUParser(), BasicGELUBindings)
GEMMMapper = NodeMapper(GenericGEMMParser(), BasicGEMMBindings)
LayerNormMapper = NodeMapper(LayerNormParser(), BasicLayerNormBindings)
iLayerNormMapper = NodeMapper(iLayerNormParser(), BasicLayerNormBindings)
DivMapper = NodeMapper(DivParser(), BasicDivBindings)
IntegerDivMapper = NodeMapper(IntegerDivParser(), BasicDivBindings)
ITAMaxMapper = NodeMapper(ITAMaxParser(), [BasicITASoftmaxBinding])
ITAPartialMaxMapper = NodeMapper(ITAPartialMaxParser(), [BasicITAPartialSoftmaxBinding])
MatMulMapper = NodeMapper(MatMulParser(), BasicMatMulBindings)
MaxPool2DMapper = NodeMapper(GenericMaxPool2DParser(), BasicMaxPool2DBindings)
MaxPool1DMapper = NodeMapper(MaxPool1DParser(), BasicMaxPool1DBindings)
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
PowMapper = NodeMapper(PowParser(), BasicPowBindings)
SqrtMapper = NodeMapper(SqrtParser(), BasicSqrtBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
ReduceSumMapper = NodeMapper(ReduceSumParser(), BasicReduceSumBindings)
ReluMapper = NodeMapper(ReluParser(), [BasicReluBinding])
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELUMapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
SoftmaxMapper = NodeMapper(SoftmaxParser(), BasicSoftmaxBindings)
iSoftmaxMapper = NodeMapper(iSoftmaxParser(), BasicSoftmaxBindings)
TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)
QuantMapper = NodeMapper(QuantParser(), BasicQuantBindings)
DequantMapper = NodeMapper(DequantParser(), BasicDequantBindings)
BatchNormalizationMapper = NodeMapper(BatchNormParser(), BasicBatchNormBindings)
ConvTransposeMapper = NodeMapper(ConvTranspose1DParser(), BasicConvTransposeBindings)
SliceMapper = NodeMapper(SliceParser(), BasicSliceBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

GenericMapping = {
    'Add': AddLayer([AddMapper]),
    'Conv': ConvLayer([Conv2DMapper, DWConv2DMapper, Conv1DMapper, DWConv1DMapper]),
    'Concat': ConcatLayer([ConcatMapper]),
    'DebugPrint': DebugPrintLayer([DebugMapper]),
    'Div': DivLayer([DivMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Gemm': GEMMLayer([GEMMMapper]),
    'iGELU': GELULayer([GELUMapper]),
    'Gelu': GELULayer([GELUMapper]),
    'LayerNormalization': LayerNormLayer([LayerNormMapper]),
    'iLayerNorm': LayerNormLayer([iLayerNormMapper]),
    'IntegerDiv': DivLayer([IntegerDivMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'Softmax': SoftmaxLayer([SoftmaxMapper]),
    'iSoftmax': SoftmaxLayer([iSoftmaxMapper]),
    'ITAMax': ITAMaxLayer([ITAMaxMapper]),
    'ITAPartialMax': ITAMaxLayer([ITAPartialMaxMapper]),
    'MatMul': GEMMLayer([MatMulMapper]),
    'MatMulInteger': MatMulLayer([MatMulMapper]),
    'MaxPool': MaxPoolLayer([MaxPool1DMapper, MaxPool2DMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pow': PowLayer([PowMapper]),
    'Sqrt': SqrtLayer([SqrtMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'ReduceSum': ReduceSumLayer([ReduceSumMapper]),
    'Relu': ReluLayer([ReluMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELUMapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Squeeze': ReshapeLayer([UnsqueezeMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Slice': SliceLayer([SliceMapper]),
    'Quant': QuantLayer([QuantMapper]),
    'Dequant': DequantLayer([DequantMapper]),
    'BatchNormalization': BatchNormalizationLayer([BatchNormalizationMapper]),
    'ConvTranspose': ConvTransposeLayer([ConvTransposeMapper])
    # # For example, you can use the DummpyMapper, in case you want to test
    # # deployment or optimizations with GlobalAveragePool nodes but did not yet
    # # implement the corresponding kernel
    # 'GlobalAveragePool': ConvLayer([DummyMapper]),
}


class GenericVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class GenericTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.referenceInitTemplate
    allocTemplate = AllocateTemplate.referenceAllocateTemplate
    deallocTemplate = FreeTemplate.referenceLocalTemplate


class GenericConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.referenceGlobalInitTemplate
    allocTemplate = AllocateTemplate.referenceGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.referenceGlobalTemplate


class GenericStructBuffer(StructBuffer):

    initTemplate = AllocateTemplate.referenceStructInitTemplate
    allocTemplate = AllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


GenericOptimizer = TopologyOptimizer(
    [
        QuantPatternPass(),
        DequantPatternPass(),
        iGELURequantMergePass(),
        MatMulAddMergePass(),
        MergeConstAddAndRequantPass(),
        ExtractPaddingFromConvPass(),
        ExtractPaddingFromPoolPass(),
        RemoveEmptyConvBiasPass(),
        RemoveOnlySingletonReduceMeanPass(),
        # DebugPrintPass(r'.*[Mm]at[Mm]ul.*', position = 'after'),
    ],
    name = "GenericOptimizer")

includeList = ["DeeployBasicMath.h"]


class GenericEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = GenericMapping, initCode: str = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class GenericPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [GenericEngine("Generic")],
                 variableBuffer = GenericVariableBuffer,
                 constantBuffer = GenericConstantBuffer,
                 structBuffer = GenericStructBuffer,
                 transientBuffer = GenericTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
