# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import itertools

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration, MemoryPassthroughGeneration
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, SignedIntegerDataTypes, float32_t, \
    int8_t, int32_t, int64_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import AddTemplate, AndTemplate, BatchNormalizationTemplate, CastTemplate, \
    ComparisonTemplate, ConcatTemplate, ConvTemplate, ConvTransposeTemplate, CosTemplate, DebugPrintTemplate, \
    DequantTemplate, DummyTemplate, DWConvTemplate, ExpandTemplate, FloatAddTemplate, FloatAveragePoolTemplate, \
    FloatCeilTemplate, FloatClipTemplate, FloatConvTemplate, FloatDivTemplate, FloatDWConvTemplate, FloatErfTemplate, \
    FloatExpandTemplate, FloatExpTemplate, FloatFloorTemplate, FloatGatherElementsTemplate, FloatGELUTemplate, \
    FloatGemmTemplate, FloatGlobalAveragePoolTemplate, FloatGlobalMaxPoolTemplate, FloatGridSampleTemplate, \
    FloatGroupNormTemplate, FloatHardSigmoidTemplate, FloatHardSwishTemplate, FloatInstanceNormTemplate, \
    FloatLayernormTemplate, FloatMatMulTemplate, FloatMaxPoolTemplate, FloatModTemplate, FloatMulTemplate, \
    FloatPadTemplate, FloatPowTemplate, FloatReduceMeanTemplate, FloatReluTemplate, FloatResizeTemplate, \
    FloatScatterNDTemplate, FloatSigmoidTemplate, FloatSoftmaxTemplate, FloatSplitTemplate, FloatSqrtTemplate, \
    FloatSubTemplate, FloatSwishTemplate, FloatTileTemplate, FloatTopKTemplate, FloatWhereTemplate, GatherTemplate, \
    GemmTemplate, IntegerDivTemplate, IsInfTemplate, IsNanTemplate, ITAMaxTemplate, ITAPartialMaxTemplate, \
    LogTemplate, MatMulTemplate, MaxPoolTemplate, ModTemplate, MulTemplate, NotTemplate, PadTemplate, QuantTemplate, \
    ReduceMaxTemplate, ReduceMeanTemplate, ReduceSumTemplate, RequantShiftTemplate, ReshapeTemplate, \
    RQIntegerDivTemplate, RQSiGELUTemplate, ScatterNDTemplate, SinTemplate, SliceTemplate, SubTemplate, TanhTemplate, \
    TileTemplate, TransposeTemplate, iGELUTemplate, iLayernormTemplate, iRMSNormTemplate, iSoftmaxTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, AndChecker, BatchNormChecker, CastChecker, ConcatChecker, \
    ConvChecker, CosChecker, DebugPrintChecker, DequantChecker, DivChecker, DummyChecker, EqualChecker, ErfChecker, \
    ExpandChecker, GatherChecker, GatherElementsChecker, GELUChecker, GEMMChecker, GreaterChecker, GridSampleChecker, \
    IdentityCastChecker, IsInfChecker, IsNanChecker, LayerNormChecker, LogChecker, MatMulChecker, MaxPoolChecker, \
    ModChecker, MulChecker, NotChecker, PadChecker, QuantChecker, ReduceMaxChecker, ReduceMeanChecker, \
    ReduceSumChecker, ReluChecker, RequantShiftChecker, ReshapeChecker, ResizeChecker, RQIntegerDivChecker, \
    ScatterNDChecker, SinChecker, SliceChecker, SoftmaxChecker, SplitChecker, TanhChecker, TileChecker, TopKChecker, \
    TransposeChecker, WhereChecker

BasicTransformer = CodeTransformation([ArgumentStructGeneration(), MemoryManagementGeneration(), FutureGeneration()])

ReshapeSkipTransformer = CodeTransformation(
    [ArgumentStructGeneration(), MemoryPassthroughGeneration(),
     FutureGeneration()])

BasicSliceBindings = [
    NodeBinding(
        SliceChecker([
            PointerClass(type),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t)
        ], [PointerClass(type)]), SliceTemplate.referenceTemplate, BasicTransformer)
    for type in (*FloatDataTypes, *IntegerDataTypes)
]

BasicAddBindings = [
    NodeBinding(AddChecker([PointerClass(type1), PointerClass(type2)], [PointerClass(int32_t)]),
                AddTemplate.referenceTemplate, BasicTransformer)
    for type1 in IntegerDataTypes
    for type2 in IntegerDataTypes
] + [
    NodeBinding(AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAddTemplate.referenceTemplate, BasicTransformer)
]

# using AddChecker since they are exactly the same
BasicSubBindings = [
    NodeBinding(AddChecker([PointerClass(type1), PointerClass(type2)], [PointerClass(int32_t)]),
                SubTemplate.referenceTemplate, BasicTransformer)
    for type1 in IntegerDataTypes
    for type2 in IntegerDataTypes
] + [
    NodeBinding(AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSubTemplate.referenceTemplate, BasicTransformer)
]

BasicConv1DBindings = [
    NodeBinding(ConvChecker(
        [PointerClass(type), PointerClass(type), PointerClass(type)], [PointerClass(type)]),
                FloatConvTemplate.reference1DTemplate, BasicTransformer) for type in FloatDataTypes
] + [
    NodeBinding(ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                ConvTemplate.reference1DTemplate, BasicTransformer)
]

BasicDWConv1DBinding = NodeBinding(ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                                   DWConvTemplate.reference1DTemplate, BasicTransformer)

BasicConv2DBindings = [
    NodeBinding(ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                ConvTemplate.reference2DTemplate, BasicTransformer)
] + [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatConvTemplate.reference2DTemplate,
        BasicTransformer)
]

BasicDWConv2DBindings = [
    NodeBinding(ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                DWConvTemplate.reference2DTemplate, BasicTransformer)
] + [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatDWConvTemplate.reference2DTemplate,
        BasicTransformer)
]

BasicDebugPrintBindings = [
    NodeBinding(DebugPrintChecker([PointerClass(type)], [PointerClass(type)]), DebugPrintTemplate.referenceTemplate,
                ReshapeSkipTransformer) for type in SignedIntegerDataTypes
]

BasicGatherBindings = [
    NodeBinding(GatherChecker([PointerClass(type), PointerClass(int32_t)], [PointerClass(type)]),
                GatherTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes
] + [
    NodeBinding(GatherChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
                GatherTemplate.referenceTemplate, BasicTransformer) for type in IntegerDataTypes
]

BasicGELUBindings = [
    NodeBinding(GELUChecker([PointerClass(int8_t)], [PointerClass(int32_t)]), iGELUTemplate.referenceTemplate,
                BasicTransformer)
] + [
    NodeBinding(GELUChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGELUTemplate.referenceTemplate,
                BasicTransformer)
]

BasicGEMMBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(int8_t), PointerClass(int8_t),
                     PointerClass(int32_t)], [PointerClass(int32_t)]), GemmTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(
        GEMMChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGemmTemplate.referenceTemplate,
        BasicTransformer)
]

BasicPowBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatPowTemplate.referenceTemplate, BasicTransformer),
]

BasicSqrtBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatSqrtTemplate.referenceTemplate,
                BasicTransformer),
]

BasicDivBindings = [
    NodeBinding(DivChecker([PointerClass(int32_t), PointerClass(int32_t)], [PointerClass(int32_t)]),
                IntegerDivTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(DivChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatDivTemplate.referenceTemplate, BasicTransformer)
]

BasicITASoftmaxBinding = NodeBinding(SoftmaxChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
                                     ITAMaxTemplate.referenceTemplate, BasicTransformer)

BasicITAPartialSoftmaxBinding = NodeBinding(SoftmaxChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
                                            ITAPartialMaxTemplate.referenceTemplate, BasicTransformer)

BasicLayerNormBindings = [
    NodeBinding(
        LayerNormChecker([PointerClass(int8_t), PointerClass(int32_t),
                          PointerClass(int32_t)], [PointerClass(int8_t)]), iLayernormTemplate.referenceTemplate,
        BasicTransformer)
] + [
    NodeBinding(
        LayerNormChecker(
            [PointerClass(float32_t), PointerClass(float32_t),
             PointerClass(float32_t)],
            [PointerClass(float32_t), PointerClass(float32_t),
             PointerClass(float32_t)]), FloatLayernormTemplate.referenceTemplate, BasicTransformer)
]

BasicMatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                MatMulTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMatMulTemplate.referenceTemplate, BasicTransformer)
]

BasicMaxPool1DBindings = [
    NodeBinding(MaxPoolChecker([PointerClass(int8_t)], [PointerClass(int8_t)]), MaxPoolTemplate.reference1DTemplate,
                BasicTransformer)
] + [
    NodeBinding(MaxPoolChecker([PointerClass(type)], [PointerClass(type)]), FloatMaxPoolTemplate.reference1DTemplate,
                BasicTransformer) for type in FloatDataTypes
]

BasicMaxPool2DBindings = [
    NodeBinding(MaxPoolChecker([PointerClass(int8_t)], [PointerClass(int8_t)]), MaxPoolTemplate.reference2DTemplate,
                BasicTransformer)
] + [
    NodeBinding(MaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMaxPoolTemplate.referenceTemplate, BasicTransformer)
]

BasicMulBindings = [
    NodeBinding(MulChecker([PointerClass(typeA), PointerClass(typeB)], [PointerClass(int32_t)]),
                MulTemplate.referenceTemplate, BasicTransformer)
    for typeA, typeB in itertools.product(SignedIntegerDataTypes, SignedIntegerDataTypes)
] + [
    NodeBinding(MulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMulTemplate.referenceTemplate, BasicTransformer)
]

BasicPad1DBindings = [
    NodeBinding(PadChecker([PointerClass(type)], [PointerClass(type)]), PadTemplate.reference1DTemplate,
                BasicTransformer) for type in SignedIntegerDataTypes
] + [
    NodeBinding(PadChecker([PointerClass(type)], [PointerClass(type)]), FloatPadTemplate.reference1DTemplate,
                BasicTransformer) for type in FloatDataTypes
]

BasicPad2DBindings = [
    NodeBinding(PadChecker([PointerClass(type)], [PointerClass(type)]), PadTemplate.reference2DTemplate,
                BasicTransformer) for type in SignedIntegerDataTypes
] + [
    NodeBinding(
        PadChecker([PointerClass(float32_t), PointerClass(float32_t),
                    PointerClass(float32_t)], [PointerClass(float32_t)]), FloatPadTemplate.reference2DTemplate,
        BasicTransformer)
]

BasicReduceMeanBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(type)], [PointerClass(type)]), ReduceMeanTemplate.referenceTemplate,
                BasicTransformer) for type in SignedIntegerDataTypes
] + [
    # ONNX OPSET < 18
    NodeBinding(ReduceMeanChecker([PointerClass(float_type), PointerClass(integer_type)], [PointerClass(float_type)]),
                FloatReduceMeanTemplate.referenceTemplate, BasicTransformer)
    for integer_type in SignedIntegerDataTypes
    for float_type in FloatDataTypes
] + [
    # ONNX OPSET >= 18
    NodeBinding(ReduceMeanChecker([PointerClass(float_type)], [PointerClass(float_type)]),
                FloatReduceMeanTemplate.referenceTemplate, BasicTransformer) for float_type in FloatDataTypes
]

BasicReduceSumBindings = [
    NodeBinding(ReduceSumChecker([PointerClass(type)], [PointerClass(int32_t)]), ReduceSumTemplate.referenceTemplate,
                BasicTransformer) for type in SignedIntegerDataTypes
]

BasicReluBinding = NodeBinding(ReluChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                               FloatReluTemplate.referenceTemplate, BasicTransformer)

BasicReshapeBindings = [
    NodeBinding(ReshapeChecker([PointerClass(type), PointerClass(int32_t)], [PointerClass(type)]),
                ReshapeTemplate.referenceTemplate, ReshapeSkipTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(ReshapeChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
                ReshapeTemplate.referenceTemplate, ReshapeSkipTransformer) for type in IntegerDataTypes
]

BasicRQSBindings = [
    NodeBinding(
        RequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                             PointerClass(int32_t)], [PointerClass(int8_t)]), RequantShiftTemplate.referenceTemplate,
        BasicTransformer) for type in SignedIntegerDataTypes
]

BasicRQSGELUBinding = NodeBinding(
    GELUChecker([PointerClass(int8_t),
                 PointerClass(int32_t),
                 PointerClass(int32_t),
                 PointerClass(int32_t)], [PointerClass(int8_t)]), RQSiGELUTemplate.referenceTemplate, BasicTransformer)

BasicRQIntegerDivBinding = NodeBinding(
    RQIntegerDivChecker([
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t)
    ], [PointerClass(int8_t)]), RQIntegerDivTemplate.referenceTemplate, BasicTransformer)

BasicSoftmaxBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(int8_t)], [PointerClass(int8_t)]), iSoftmaxTemplate.referenceTemplate,
                BasicTransformer)
] + [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSoftmaxTemplate.referenceTemplate, BasicTransformer)
]

BasicTransposeBindings = [
    NodeBinding(TransposeChecker([PointerClass(type)], [PointerClass(type)]), TransposeTemplate.referenceTemplate,
                BasicTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(TransposeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                TransposeTemplate.referenceTemplate, BasicTransformer)
]

BasiciRMSNormBinding = NodeBinding(
    LayerNormChecker([PointerClass(int8_t), PointerClass(int32_t)], [PointerClass(int8_t)]),
    iRMSNormTemplate.referenceTemplate, BasicTransformer)

DummyBinding = NodeBinding(DummyChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
                           DummyTemplate.referenceTemplate, BasicTransformer)

BasicConcatBindings = [
    NodeBinding(ConcatChecker([PointerClass(type), PointerClass(type)], [PointerClass(type)]),
                ConcatTemplate.referenceTemplate, BasicTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(ConcatChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                ConcatTemplate.referenceTemplate, BasicTransformer)
]

BasicQuantBindings = [
    NodeBinding(QuantChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), QuantTemplate.referenceTemplate,
                BasicTransformer),
]

BasicDequantBindings = [
    NodeBinding(DequantChecker([PointerClass(int8_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                BasicTransformer),
] + [
    NodeBinding(DequantChecker([PointerClass(int32_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                BasicTransformer),
]

BasicBatchNormBindings = [
    NodeBinding(
        BatchNormChecker(
            [PointerClass(type),
             PointerClass(type),
             PointerClass(type),
             PointerClass(type),
             PointerClass(type)], [PointerClass(type)]), BatchNormalizationTemplate.referenceTemplate, BasicTransformer)
    for type in FloatDataTypes
]

BasicConvTransposeBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(type), PointerClass(type), PointerClass(type)],  # input, weight, bias
            [PointerClass(type)]),
        ConvTransposeTemplate.referenceTemplate,
        BasicTransformer) for type in FloatDataTypes
] + [
    NodeBinding(
        ConvChecker(
            [PointerClass(type), PointerClass(type)],  # input, weight
            [PointerClass(type)]),
        ConvTransposeTemplate.referenceTemplate,
        BasicTransformer) for type in FloatDataTypes
]

BasicCeilBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatCeilTemplate.referenceTemplate,
                BasicTransformer),
]

BasicFloorBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatFloorTemplate.referenceTemplate, BasicTransformer),
]

BasicClipBindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float32_t), PointerClass(float32_t),
             PointerClass(float32_t)], [PointerClass(float32_t)]), FloatClipTemplate.referenceTemplate,
        BasicTransformer),
]

BasicExpBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatExpTemplate.referenceTemplate,
                BasicTransformer),
]

BasicSigmoidBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSigmoidTemplate.referenceTemplate, BasicTransformer),
]

BasicSwishBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSwishTemplate.referenceTemplate, BasicTransformer),
]

BasicHardSigmoidBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatHardSigmoidTemplate.referenceTemplate, BasicTransformer),
]

BasicHardSwishBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatHardSwishTemplate.referenceTemplate, BasicTransformer),
]

BasicInstanceNormBindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float32_t), PointerClass(float32_t),
             PointerClass(float32_t)], [PointerClass(float32_t)]), FloatInstanceNormTemplate.referenceTemplate,
        BasicTransformer),
]

BasicGroupNormBindings = [
    NodeBinding(
        DummyChecker(
            [PointerClass(float32_t), PointerClass(float32_t),
             PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGroupNormTemplate.referenceTemplate,
        BasicTransformer),
]

BasicAveragePool1DBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAveragePoolTemplate.referenceTemplate1d, BasicTransformer)
]

BasicAveragePool2DBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAveragePoolTemplate.referenceTemplate2d, BasicTransformer)
]

BasicGlobalAveragePoolBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatGlobalAveragePoolTemplate.referenceTemplate, BasicTransformer)
]

BasicGlobalMaxPoolBindings = [
    NodeBinding(DummyChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatGlobalMaxPoolTemplate.referenceTemplate, BasicTransformer)
]

BasicTanhBindings = [
    NodeBinding(TanhChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), TanhTemplate.referenceTemplate,
                BasicTransformer)
]

BasicReduceMaxBindings = [
    NodeBinding(ReduceMaxChecker([PointerClass(type)], [PointerClass(int32_t)]), ReduceMaxTemplate.referenceTemplate,
                BasicTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(ReduceMaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                ReduceMaxTemplate.referenceTemplate, BasicTransformer)
]
BasicGreaterBindings = [
    NodeBinding(GreaterChecker([PointerClass(type1), PointerClass(type2)], [PointerClass(int32_t)]),
                ComparisonTemplate.referenceTemplate, BasicTransformer)
    for type1 in IntegerDataTypes
    for type2 in IntegerDataTypes
] + [
    NodeBinding(GreaterChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(int32_t)]),
                ComparisonTemplate.referenceTemplate, BasicTransformer)
]

BasicEqualBinding = [
    NodeBinding(EqualChecker([PointerClass(type1), PointerClass(type2)], [PointerClass(int8_t)]),
                ComparisonTemplate.referenceTemplate, BasicTransformer)
    for type1 in FloatDataTypes
    for type2 in FloatDataTypes
]

BasicAndBindings = [
    NodeBinding(AndChecker([PointerClass(type1), PointerClass(type2)], [PointerClass(int8_t)]),
                AndTemplate.referenceTemplate, BasicTransformer)
    for type1 in IntegerDataTypes
    for type2 in IntegerDataTypes
]

BasicIsInfBindings = [
    NodeBinding(IsInfChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), IsInfTemplate.referenceTemplate,
                BasicTransformer)
]

BasicNotBindings = [
    NodeBinding(NotChecker([PointerClass(_type)], [PointerClass(int8_t)]), NotTemplate.referenceTemplate,
                BasicTransformer) for _type in IntegerDataTypes
]

BasicIsNanBindings = [
    NodeBinding(IsNanChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), IsNanTemplate.referenceTemplate,
                BasicTransformer)
]

BasicScatterNDBindings = [
    NodeBinding(
        ScatterNDChecker(
            [PointerClass(float32_t), PointerClass(int64_t),
             PointerClass(float32_t)], [PointerClass(float32_t)]), FloatScatterNDTemplate.referenceTemplate,
        BasicTransformer)
] + [
    NodeBinding(
        ScatterNDChecker([PointerClass(int8_t), PointerClass(int64_t),
                          PointerClass(int8_t)], [PointerClass(int8_t)]), ScatterNDTemplate.referenceTemplate,
        BasicTransformer)
]

BasicCosBindings = [
    NodeBinding(CosChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), CosTemplate.referenceTemplate,
                BasicTransformer)
]

BasicSinBindings = [
    NodeBinding(SinChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), SinTemplate.referenceTemplate,
                BasicTransformer)
]

BasicExpandBindings = [
    NodeBinding(ExpandChecker([PointerClass(float32_t), PointerClass(int64_t)], [PointerClass(float32_t)]),
                FloatExpandTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(ExpandChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int8_t)]),
                ExpandTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(ExpandChecker([PointerClass(int64_t), PointerClass(int64_t)], [PointerClass(int64_t)]),
                ExpandTemplate.referenceTemplate, BasicTransformer)
]

BasicGatherElementsBindings = [
    NodeBinding(GatherElementsChecker([PointerClass(float32_t), PointerClass(int32_t)], [PointerClass(float32_t)]),
                FloatGatherElementsTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(GatherElementsChecker([PointerClass(float32_t), PointerClass(int64_t)], [PointerClass(float32_t)]),
                FloatGatherElementsTemplate.referenceTemplate, BasicTransformer)
]

BasicTileBindings = [
    NodeBinding(TileChecker([PointerClass(float32_t), PointerClass(int64_t)], [PointerClass(float32_t)]),
                FloatTileTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(TileChecker([PointerClass(int64_t), PointerClass(int64_t)], [PointerClass(int64_t)]),
                TileTemplate.referenceTemplate, BasicTransformer)
]

BasicTopKBindings = [
    NodeBinding(
        TopKChecker([PointerClass(float32_t), PointerClass(int64_t)],
                    [PointerClass(float32_t), PointerClass(int64_t)]), FloatTopKTemplate.referenceTemplate,
        BasicTransformer)
]

BasicLogBindings = [
    NodeBinding(LogChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), LogTemplate.referenceTemplate,
                BasicTransformer)
]

BasicErfBindings = [
    NodeBinding(ErfChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatErfTemplate.referenceTemplate,
                BasicTransformer)
]

BasicWhereBindings = [
    NodeBinding(
        WhereChecker([PointerClass(int8_t), PointerClass(float32_t),
                      PointerClass(float32_t)], [PointerClass(float32_t)]), FloatWhereTemplate.referenceTemplate,
        BasicTransformer)
]

BasicModBindings = [
    NodeBinding(ModChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatModTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(ModChecker([PointerClass(int64_t), PointerClass(int64_t)], [PointerClass(int64_t)]),
                ModTemplate.referenceTemplate, BasicTransformer)
]

BasicResizeBindings = [
    NodeBinding(
        ResizeChecker(
            [PointerClass(float32_t), PointerClass(float32_t),
             PointerClass(float32_t)], [PointerClass(float32_t)]), FloatResizeTemplate.referenceTemplate,
        BasicTransformer)
]

BasicGridSampleBindings = [
    NodeBinding(GridSampleChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatGridSampleTemplate.referenceTemplate, BasicTransformer)
]

BasicCastBindings = [
    NodeBinding(
        CastChecker([PointerClass(int64_t)],
                    [PointerClass(float32_t)]),  # https://app.semanticdiff.com/gh/onnx/onnx-mlir/pull/2486/overview
        CastTemplate.referenceTemplate,
        BasicTransformer)  #NOT ReshapeSkipTransformer) othrerwise, Identity Nodes appear since shape does not change
] + [
    NodeBinding(CastChecker([PointerClass(typeA)], [PointerClass(typeB)]), CastTemplate.referenceTemplate,
                BasicTransformer) for typeA, typeB in itertools.product(IntegerDataTypes, IntegerDataTypes)
] + [
    NodeBinding(CastChecker([PointerClass(float32_t)], [PointerClass(int64_t)]), CastTemplate.referenceTemplate,
                BasicTransformer)
]

BasicIdentityCastBindings = [
    NodeBinding(IdentityCastChecker([PointerClass(int64_t)], [PointerClass(int64_t)]), CastTemplate.referenceTemplate,
                BasicTransformer)
] + [
    NodeBinding(IdentityCastChecker([PointerClass(typeA)], [PointerClass(typeB)]), CastTemplate.referenceTemplate,
                BasicTransformer)  #SAME TEMPLATE AS CAST
    for typeA, typeB in itertools.product(IntegerDataTypes, IntegerDataTypes)
]

BasicSplitBindings = [  #ORDER MATTERS
    NodeBinding(
        SplitChecker(
            [PointerClass(float32_t), PointerClass(int64_t)],
            [PointerClass(float32_t), PointerClass(float32_t),
             PointerClass(float32_t)]), FloatSplitTemplate.referenceTemplate, BasicTransformer)
] + [
    NodeBinding(
        SplitChecker([PointerClass(float32_t), PointerClass(int64_t)],
                     [PointerClass(float32_t), PointerClass(float32_t)]), FloatSplitTemplate.referenceTemplate,
        BasicTransformer)
] + [
    NodeBinding(SplitChecker([PointerClass(float32_t), PointerClass(int64_t)], [PointerClass(float32_t)]),
                FloatSplitTemplate.referenceTemplate, BasicTransformer)
]
