# ----------------------------------------------------------------------
#
# File: BasicBindings.py
#
# Last edited: 17.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author:
# - Moritz Scherer, ETH Zurich
# - Philip Wiese, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration, MemoryPassthroughGeneration
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes, SignedIntegerDataTypes, float32_t, int8_t, int32_t, \
    uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import AddTemplate, ConcatTemplate, ConvTemplate, DebugPrintTemplate, \
    DummyTemplate, DWConvTemplate, FloatAddTemplate, FloatConvTemplate, FloatDivTemplate, FloatGemmTemplate, \
    FloatLayernormTemplate, FloatReluTemplate, FloatSoftmaxTemplate, GatherTemplate, GemmTemplate, IntegerDivTemplate, \
    ITAMaxTemplate, ITAPartialMaxTemplate, MatMulTemplate, MaxPoolTemplate, MulTemplate, PadTemplate, \
    ReduceMeanTemplate, ReduceSumTemplate, RequantShiftTemplate, ReshapeTemplate, RQIntegerDivTemplate, \
    RQSiGELUTemplate, SliceTemplate, TransposeTemplate, iGELUTemplate, iLayernormTemplate, iRMSNormTemplate, \
    iSoftmaxTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, ConcatChecker, ConvChecker, DebugPrintChecker, \
    DivChecker, DummyChecker, GatherChecker, GELUChecker, GEMMChecker, LayerNormChecker, MatMulChecker, \
    MaxPoolChecker, MulChecker, PadChecker, ReduceMeanChecker, ReduceSumChecker, ReluChecker, RequantShiftChecker, \
    ReshapeChecker, RQIntegerDivChecker, SliceChecker, SoftmaxChecker, TransposeChecker

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
        ], [PointerClass(type)]), SliceTemplate.referenceTemplate, BasicTransformer) for type in IntegerDataTypes
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

BasicConv1DBinding = NodeBinding(ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                                 ConvTemplate.reference1DTemplate, BasicTransformer)

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

BasicDWConv2DBinding = NodeBinding(ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                                   DWConvTemplate.reference2DTemplate, BasicTransformer)

BasicDebugPrintBindings = [
    NodeBinding(DebugPrintChecker([PointerClass(type)], [PointerClass(type)]), DebugPrintTemplate.referenceTemplate,
                ReshapeSkipTransformer) for type in SignedIntegerDataTypes
]

BasicGatherBindings = [
    NodeBinding(GatherChecker([PointerClass(type), PointerClass(int32_t)], [PointerClass(type)]),
                GatherTemplate.referenceTemplate, BasicTransformer) for type in SignedIntegerDataTypes
]

BasicGELUBinding = NodeBinding(GELUChecker([PointerClass(int8_t)], [PointerClass(int32_t)]),
                               iGELUTemplate.referenceTemplate, BasicTransformer)

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
             PointerClass(float32_t)], [PointerClass(float32_t)]), FloatLayernormTemplate.referenceTemplate,
        BasicTransformer)
]

BasicMatMulBinding = NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                                 MatMulTemplate.referenceTemplate, BasicTransformer)

BasicMaxPool2DBinding = NodeBinding(MaxPoolChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
                                    MaxPoolTemplate.referenceTemplate, BasicTransformer)

BasicMulBindings = [
    NodeBinding(MulChecker([PointerClass(typeA), PointerClass(typeB)], [PointerClass(int32_t)]),
                MulTemplate.referenceTemplate, BasicTransformer)
    for typeA, typeB in itertools.product(SignedIntegerDataTypes, SignedIntegerDataTypes)
]

BasicPad1DBindings = [
    NodeBinding(PadChecker([PointerClass(type)], [PointerClass(type)]), PadTemplate.reference1DTemplate,
                BasicTransformer) for type in SignedIntegerDataTypes
]
BasicPad2DBindings = [
    NodeBinding(PadChecker([PointerClass(type)], [PointerClass(type)]), PadTemplate.reference2DTemplate,
                BasicTransformer) for type in SignedIntegerDataTypes
] + [
    NodeBinding(
        PadChecker([PointerClass(float32_t), PointerClass(float32_t),
                    PointerClass(float32_t)], [PointerClass(float32_t)]), PadTemplate.reference2DTemplate,
        BasicTransformer)
]

BasicReduceMeanBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(type)], [PointerClass(type)]), ReduceMeanTemplate.referenceTemplate,
                BasicTransformer) for type in SignedIntegerDataTypes
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
    NodeBinding(ReshapeChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                ReshapeTemplate.referenceTemplate, ReshapeSkipTransformer)
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
