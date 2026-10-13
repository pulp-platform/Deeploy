# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.CommonExtensions.OptimizationPasses.TopologyOptimizationPasses.LoweringOptimizationPasses import \
    RemoveEmptyConvBiasPass, RemoveOnlySingletonReduceMeanPass
from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicAndBindings, BasicAveragePool1DBindings, \
    BasicAveragePool2DBindings, BasicBatchNormBindings, BasicCastBindings, BasicCeilBindings, BasicClipBindings, \
    BasicConcatBindings, BasicConv1DBindings, BasicConv2DBindings, BasicConvTransposeBindings, BasicCosBindings, \
    BasicDebugPrintBindings, BasicDequantBindings, BasicDivBindings, BasicDWConv1DBinding, BasicDWConv2DBindings, \
    BasicEqualBinding, BasicErfBindings, BasicExpandBindings, BasicExpBindings, BasicFloorBindings, \
    BasicGatherBindings, BasicGatherElementsBindings, BasicGELUBindings, BasicGEMMBindings, \
    BasicGlobalAveragePoolBindings, BasicGlobalMaxPoolBindings, BasicGreaterBindings, BasicGridSampleBindings, \
    BasicGroupNormBindings, BasicHardSigmoidBindings, BasicHardSwishBindings, BasicIdentityCastBindings, \
    BasicInstanceNormBindings, BasicIsInfBindings, BasicIsNanBindings, BasicITAPartialSoftmaxBinding, \
    BasicITASoftmaxBinding, BasicLayerNormBindings, BasicLogBindings, BasicMatMulBindings, BasicMaxPool1DBindings, \
    BasicMaxPool2DBindings, BasicModBindings, BasicMulBindings, BasicNotBindings, BasicPad1DBindings, \
    BasicPad2DBindings, BasicPowBindings, BasicQuantBindings, BasicReduceMaxBindings, BasicReduceMeanBindings, \
    BasicReduceSumBindings, BasicReluBinding, BasicReshapeBindings, BasicResizeBindings, BasicRQIntegerDivBinding, \
    BasicRQSBindings, BasicRQSGELUBinding, BasicScatterNDBindings, BasicSigmoidBindings, BasicSinBindings, \
    BasicSliceBindings, BasicSoftmaxBindings, BasicSplitBindings, BasicSqrtBindings, BasicSubBindings, \
    BasicSwishBindings, BasicTanhBindings, BasicTileBindings, BasicTopKBindings, BasicTransposeBindings, \
    BasicWhereBindings, DummyBinding
from Deeploy.Targets.Generic.Layers import AddLayer, AndLayer, AveragePoolLayer, BatchNormalizationLayer, CastLayer, \
    CeilLayer, ClipLayer, ComparisonLayer, ConcatLayer, ConvLayer, ConvTransposeLayer, CosLayer, DebugPrintLayer, \
    DequantLayer, DivLayer, ErfLayer, ExpandLayer, ExpLayer, FloorLayer, GatherElementsLayer, GatherLayer, GELULayer, \
    GEMMLayer, GlobalAveragePoolLayer, GlobalMaxPoolLayer, GridSampleLayer, GroupNormLayer, InstanceNormLayer, \
    IsInfLayer, IsNanLayer, ITAMaxLayer, LayerNormLayer, LogLayer, MatMulLayer, MaxPoolLayer, ModLayer, \
    MulLayer, NotLayer, PadLayer, PowLayer, QuantLayer, ReduceMaxLayer, ReduceMeanLayer, ReduceSumLayer, ReluLayer, \
    RequantShiftLayer, ReshapeLayer, ResizeLayer, RQIntegerDivLayer, RQSiGELULayer, ScatterNDLayer, SigmoidLayer, \
    SinLayer, SliceLayer, SoftmaxLayer, SplitLayer, SqrtLayer, SubLayer, SwishLayer, TanhLayer, TileLayer, \
    TopKLayer, TransposeLayer, WhereLayer
from Deeploy.Targets.Generic.Parsers import AddParser, AndParser, AveragePool1DParser, AveragePool2DParser, \
    BatchNormParser, CastParser, CeilParser, ClipParser, ConcatParser, ConvTranspose1DParser, CosParser, DebugParser, \
    DequantParser, DivParser, DummyParser, EqualParser, ErfParser, ExpandParser, ExpParser, FlattenParser, \
    FloorParser, GatherElementsParser, GatherParser, GELUParser, GenericConv1DParser, GenericConv2DParser, \
    GenericDWConv1DParser, GenericDWConv2DParser, GenericGEMMParser, GenericMaxPool2DParser, GlobalAveragePoolParser, \
    GlobalMaxPoolParser, GreaterParser, GridSampleParser, GroupNormParser, HardSigmoidParser, HardSwishParser, \
    IdentityCastParser, InstanceNormParser, IntegerDivParser, IsInfParser, IsNanParser, ITAMaxParser, \
    ITAPartialMaxParser, LayerNormParser, LessParser, LogParser, MatMulParser, MaxPool1DParser, ModParser, MulParser, \
    NotParser, Pad1DParser, Pad2DParser, PowParser, QuantParser, ReduceMaxParser, ReduceMeanParser, ReduceSumParser, \
    ReluParser, RequantShiftParser, ReshapeParser, ResizeParser, RQIntegerDivParser, RQSiGELUParser, ScatterNDParser, \
    SigmoidParser, SinParser, SliceParser, SoftmaxParser, SplitParser, SqrtParser, SubParser, SwishParser, TanhParser, \
    TileParser, TopKParser, TransposeParser, UnsqueezeParser, WhereParser, iLayerNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import DequantPatternPass, ExtractPaddingFromConvPass, \
    ExtractPaddingFromPoolPass, MatMulAddMergePass, MergeConstAddAndRequantPass, QuantPatternPass, \
    UnrollReduceMaxPass, iGELURequantMergePass

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
SubMapper = NodeMapper(SubParser(), BasicSubBindings)
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
CeilMapper = NodeMapper(CeilParser(), BasicCeilBindings)
FloorMapper = NodeMapper(FloorParser(), BasicFloorBindings)
ClipMapper = NodeMapper(ClipParser(), BasicClipBindings)
ExpMapper = NodeMapper(ExpParser(), BasicExpBindings)
SigmoidMapper = NodeMapper(SigmoidParser(), BasicSigmoidBindings)
SwishMapper = NodeMapper(SwishParser(), BasicSwishBindings)
HardSigmoidMapper = NodeMapper(HardSigmoidParser(), BasicHardSigmoidBindings)
HardSwishMapper = NodeMapper(HardSwishParser(), BasicHardSwishBindings)
InstanceNormMapper = NodeMapper(InstanceNormParser(), BasicInstanceNormBindings)
GroupNormMapper = NodeMapper(GroupNormParser(), BasicGroupNormBindings)
AveragePool1DMapper = NodeMapper(AveragePool1DParser(), BasicAveragePool1DBindings)
AveragePool2DMapper = NodeMapper(AveragePool2DParser(), BasicAveragePool2DBindings)
GlobalAveragePoolMapper = NodeMapper(GlobalAveragePoolParser(), BasicGlobalAveragePoolBindings)
GlobalMaxPoolMapper = NodeMapper(GlobalMaxPoolParser(), BasicGlobalMaxPoolBindings)
TanhMapper = NodeMapper(TanhParser(), BasicTanhBindings)
ReduceMaxMapper = NodeMapper(ReduceMaxParser(), BasicReduceMaxBindings)
ScatterNDMapper = NodeMapper(ScatterNDParser(), BasicScatterNDBindings)
CosMapper = NodeMapper(CosParser(), BasicCosBindings)
SinMapper = NodeMapper(SinParser(), BasicSinBindings)
ExpandMapper = NodeMapper(ExpandParser(), BasicExpandBindings)
GatherElementsMapper = NodeMapper(GatherElementsParser(), BasicGatherElementsBindings)
TileMapper = NodeMapper(TileParser(), BasicTileBindings)
TopKMapper = NodeMapper(TopKParser(), BasicTopKBindings)
LogMapper = NodeMapper(LogParser(), BasicLogBindings)
ErfMapper = NodeMapper(ErfParser(), BasicErfBindings)
WhereMapper = NodeMapper(WhereParser(), BasicWhereBindings)
ModMapper = NodeMapper(ModParser(), BasicModBindings)
ResizeMapper = NodeMapper(ResizeParser(), BasicResizeBindings)
GridSampleMapper = NodeMapper(GridSampleParser(), BasicGridSampleBindings)
CastMapper = NodeMapper(CastParser(), BasicCastBindings)
IdentityCastMapper = NodeMapper(IdentityCastParser(), BasicIdentityCastBindings)
SplitMapper = NodeMapper(SplitParser(), BasicSplitBindings)
GreaterMapper = NodeMapper(GreaterParser(), BasicGreaterBindings)
LessMapper = NodeMapper(LessParser(), BasicGreaterBindings)
EqualMapper = NodeMapper(EqualParser(), BasicEqualBinding)
AndMapper = NodeMapper(AndParser(), BasicAndBindings)
IsInfMapper = NodeMapper(IsInfParser(), BasicIsInfBindings)
IsNanMapper = NodeMapper(IsNanParser(), BasicIsNanBindings)
NotMapper = NodeMapper(NotParser(), BasicNotBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

GenericMapping = {
    'Add': AddLayer([AddMapper]),
    'Sub': SubLayer([SubMapper]),
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
    'Not': NotLayer([NotMapper]),
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
    'ConvTranspose': ConvTransposeLayer([ConvTransposeMapper]),
    'Ceil': CeilLayer([CeilMapper]),
    'Floor': FloorLayer([FloorMapper]),
    'Clip': ClipLayer([ClipMapper]),
    'Exp': ExpLayer([ExpMapper]),
    'Sigmoid': SigmoidLayer([SigmoidMapper]),
    'Swish': SwishLayer([SwishMapper]),
    'HardSigmoid': SigmoidLayer([HardSigmoidMapper]),
    'HardSwish': SwishLayer([HardSwishMapper]),
    'InstanceNormalization': InstanceNormLayer([InstanceNormMapper]),
    'GroupNormalization': GroupNormLayer([GroupNormMapper]),
    'AveragePool': AveragePoolLayer([AveragePool1DMapper, AveragePool2DMapper]),
    'GlobalAveragePool': GlobalAveragePoolLayer([GlobalAveragePoolMapper]),
    'GlobalMaxPool': GlobalMaxPoolLayer([GlobalMaxPoolMapper]),
    'Tanh': TanhLayer([TanhMapper]),
    'ReduceMax': ReduceMaxLayer([ReduceMaxMapper]),
    'ScatterND': ScatterNDLayer([ScatterNDMapper]),
    'Cos': CosLayer([CosMapper]),
    'Sin': SinLayer([SinMapper]),
    'Expand': ExpandLayer([ExpandMapper]),
    'GatherElements': GatherElementsLayer([GatherElementsMapper]),
    'Tile': TileLayer([TileMapper]),
    'TopK': TopKLayer([TopKMapper]),
    'Log': LogLayer([LogMapper]),
    'Erf': ErfLayer([ErfMapper]),
    'Where': WhereLayer([WhereMapper]),
    'Mod': ModLayer([ModMapper]),
    'Resize': ResizeLayer([ResizeMapper]),
    'GridSample': GridSampleLayer([GridSampleMapper]),
    'Cast': CastLayer([CastMapper]),
    # 'Identity': IdentityCastLayer([IdentityCastMapper]),
    #SOME CASTS TURN INTO IDENTITY??
    'Split': SplitLayer([SplitMapper]),
    'Greater': ComparisonLayer([GreaterMapper]),
    'Less': ComparisonLayer([LessMapper]),
    'Equal': ComparisonLayer([EqualMapper]),
    'And': AndLayer([AndMapper]),
    'IsInf': IsInfLayer([IsInfMapper]),
    'IsNaN': IsNanLayer([IsNanMapper])
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
        UnrollReduceMaxPass(),
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
