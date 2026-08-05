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
    FloatAveragePoolTemplate, FloatCeilTemplate, FloatClipTemplate, FloatConvTemplate, FloatDivTemplate, \
    FloatDWConvTemplate, FloatExpTemplate, FloatFloorTemplate, FloatGELUTemplate, FloatGemmTemplate, \
    FloatGlobalAveragePoolTemplate, FloatGlobalMaxPoolTemplate, FloatGroupNormTemplate, FloatHardSigmoidTemplate, \
    FloatHardSwishTemplate, FloatInstanceNormTemplate, FloatLayernormTemplate, FloatMatMulTemplate, \
    FloatMaxPoolTemplate, FloatMulTemplate, FloatPadTemplate, FloatPowTemplate, FloatReduceMeanTemplate, \
    FloatReluTemplate, FloatSigmoidTemplate, FloatSoftmaxTemplate, FloatSqrtTemplate, FloatSubTemplate, \
    FloatSwishTemplate, GatherTemplate, GemmTemplate, IntegerDivTemplate, ITAMaxTemplate, ITAPartialMaxTemplate, \
    MatMulTemplate, MaxPoolTemplate, MulTemplate, PadTemplate, QuantTemplate, ReduceMeanTemplate, ReduceSumTemplate, \
    RequantShiftTemplate, ReshapeTemplate, RQIntegerDivTemplate, RQSiGELUTemplate, SliceTemplate, SubTemplate, \
    TransposeTemplate, iGELUTemplate, iLayernormTemplate, iRMSNormTemplate, iSoftmaxTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, BatchNormChecker, ConcatChecker, ConvChecker, \
    DebugPrintChecker, DequantChecker, DivChecker, DummyChecker, GatherChecker, GELUChecker, GEMMChecker, \
    LayerNormChecker, MatMulChecker, MaxPoolChecker, MulChecker, PadChecker, QuantChecker, ReduceMeanChecker, \
    ReduceSumChecker, ReluChecker, RequantShiftChecker, ReshapeChecker, RQIntegerDivChecker, SliceChecker, \
    SoftmaxChecker, TransposeChecker


