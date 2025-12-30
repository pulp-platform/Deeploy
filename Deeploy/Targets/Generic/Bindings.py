# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import itertools

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration, MemoryPassthroughGeneration
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, SignedIntegerDataTypes, float32_t, \
    int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import AddTemplate, BatchNormalizationTemplate, ConcatTemplate, ConvTemplate, \
    ConvTransposeTemplate, DebugPrintTemplate, DequantTemplate, DummyTemplate, DWConvTemplate, FloatAddTemplate, \
    FloatConvTemplate, FloatDivTemplate, FloatDWConvTemplate, FloatGELUTemplate, FloatGemmTemplate, \
    FloatLayernormTemplate, FloatMatMulTemplate, FloatMaxPoolTemplate, FloatMulTemplate, FloatPadTemplate, \
    FloatPowTemplate, FloatReduceMeanTemplate, FloatReluTemplate, FloatSoftmaxTemplate, FloatSqrtTemplate, \
    GatherTemplate, GemmTemplate, IntegerDivTemplate, ITAMaxTemplate, ITAPartialMaxTemplate, MatMulTemplate, \
    MaxPoolTemplate, MulTemplate, PadTemplate, QuantTemplate, ReduceMeanTemplate, ReduceSumTemplate, \
    RequantShiftTemplate, ReshapeTemplate, RQIntegerDivTemplate, RQSiGELUTemplate, SliceTemplate, TransposeTemplate, \
    iGELUTemplate, iLayernormTemplate, iRMSNormTemplate, iSoftmaxTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, BatchNormChecker, ConcatChecker, ConvChecker, \
    DebugPrintChecker, DequantChecker, DivChecker, DummyChecker, GatherChecker, GELUChecker, GEMMChecker, \
    LayerNormChecker, MatMulChecker, MaxPoolChecker, MulChecker, PadChecker, QuantChecker, ReduceMeanChecker, \
    ReduceSumChecker, ReluChecker, RequantShiftChecker, ReshapeChecker, RQIntegerDivChecker, SliceChecker, \
    SoftmaxChecker, TransposeChecker

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
    NodeBinding(MaxPoolChecker([PointerClass(int8_t)], [PointerClass(int8_t)]), MaxPoolTemplate.referenceTemplate,
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
