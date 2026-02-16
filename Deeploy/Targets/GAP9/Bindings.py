# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""
GAP9-specific bindings using cl_dma.h API instead of low-level MCHAN.

This module provides GAP9-specific DMA and code transformations that use
the PMSIS standard cl_dma API for better portability and cleaner code.
"""

import itertools

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration, MemoryPassthroughGeneration
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, SignedIntegerDataTypes, float32_t, \
    int8_t, int32_t, int64_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.Bindings.AutoFutureBinding import AutoFutureBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.GAP9.DMA.L3Dma import gap9L3DmaHack
from Deeploy.Targets.GAP9.DMA.MchanDma import GAP9MchanDma
# Import templates from PULPOpen and Generic
from Deeploy.Targets.Generic.Templates import AddTemplate, ConcatTemplate, DequantTemplate, FloatReduceMeanTemplate, \
    FloatReduceSumTemplate, GatherTemplate, QuantTemplate, RQSiGELUTemplate, SliceTemplate, iHardswishTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, ConcatChecker, ConvChecker, DequantChecker, \
    GatherChecker, GELUChecker, GEMMChecker, HardswishChecker, LayerNormChecker, MatMulChecker, MulChecker, \
    QuantChecker, ReduceMeanChecker, ReluChecker, ReshapeChecker, RQAddChecker, RQHardswishChecker, SGDChecker, \
    SliceChecker, SoftmaxChecker, SoftmaxCrossEntropyLossChecker, TransposeChecker
from Deeploy.Targets.PULPOpen.Bindings import ForkClosure, L3MemoryAwareFunctionCallClosure, \
    MemoryAwareForkTransformer, MemoryAwareFunctionCallClosure, TilingCallClosure
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterSynch import PULPSynchCoresPass
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterTiling import PULPClusterTiling
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPL3Tiling import PULPL3Tiling
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPProfileUntiled import PULPProfileUntiled
from Deeploy.Targets.PULPOpen.DataTypes import PULPDMAFuture
from Deeploy.Targets.PULPOpen.Templates import ConvTemplate, DMASliceTemplate, FloatAddTemplate, FloatConvTemplate, \
    FloatGELUTemplate, FloatGemmTemplate, FloatLayernormTemplate, FloatMatMulTemplate, FloatMaxPoolTemplate, \
    FloatMulTemplate, FloatReluTemplate, FloatSoftmaxTemplate, GEMMTemplate, MatrixVectorTemplate, MaxPoolTemplate, \
    MulTemplate, ReduceMeanTemplate, RequantShiftTemplate, ReshapeTemplate, RQAddTemplate, RQSiHardswishTemplate, \
    SGDTemplate, SoftmaxCrossEntropyLossTemplate, TallGEMMTemplate, TransposeTemplate, UniformRequantShiftTemplate, \
    iRMSNormTemplate, iSoftmaxTemplate
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker, PULPLinearChecker, PULPMaxPoolChecker, \
    PULPRequantShiftChecker
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement, \
    TilingVariableReplacementUpdate

# GAP9-specific transformer using cl_dma.h API
GAP9Transformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    PULPSynchCoresPass(),
    ForkClosure(writeback = False, generateStruct = True),
    TilingVariableReplacementUpdate("L1"),
    PULPClusterTiling("L2", "L1", GAP9MchanDma()),  # Use GAP9MchanDma instead of ClDma
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    TilingVariableReplacement("L2"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    PULPL3Tiling("L3", "L2", gap9L3DmaHack),  # Use GAP9-specific L3 DMA
    PULPProfileUntiled(),
    ArgumentStructGeneration(),
    L3MemoryAwareFunctionCallClosure(writeback = False),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

# GAP9-specific cluster transformer using cl_dma.h API
GAP9ClusterTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False, generateStruct = True),
    TilingVariableReplacementUpdate("L1"),
    PULPClusterTiling("L2", "L1", GAP9MchanDma()),  # Use GAP9MchanDma instead of ClDma
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    TilingVariableReplacement("L2"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    PULPL3Tiling("L3", "L2", gap9L3DmaHack),  # Use GAP9-specific L3 DMA
    PULPProfileUntiled(),
    ArgumentStructGeneration(),
    L3MemoryAwareFunctionCallClosure(writeback = False),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

# Simple transformer for non-tiling cases
GAP9SimpleTransformer = CodeTransformation([
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

# Skip transformer (no DMA operations)
GAP9SkipTransformer = CodeTransformation(
    [ArgumentStructGeneration(),
     MemoryPassthroughGeneration("L.*"),
     MemoryPassthroughGeneration(),
     FutureGeneration()])

# ===============================================================================
# GAP9-specific bindings using ClDma instead of MchanDma
# All bindings below use GAP9Transformer or GAP9ClusterTransformer
# ===============================================================================

GAP9DMASliceBindings = [
    AutoFutureBinding(
        SliceChecker([
            PointerClass(type),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t)
        ], [PULPDMAFuture(underlyingType = type)]), DMASliceTemplate.referenceTemplate, MemoryAwareForkTransformer)
    for type in IntegerDataTypes
]

GAP9SliceBindings = [
    NodeBinding(
        SliceChecker([
            PointerClass(type),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t),
            PointerClass(uint8_t)
        ], [PointerClass(type)]), SliceTemplate.referenceTemplate, GAP9Transformer) for type in FloatDataTypes
]

GAP9ReshapeBindings = [
    NodeBinding(ReshapeChecker([PointerClass(type), PointerClass(int64_t)], [PointerClass(type)]),
                ReshapeTemplate.referenceTemplate, GAP9SkipTransformer) for type in IntegerDataTypes + FloatDataTypes
]

GAP9RQAddBindings = [
    NodeBinding(RQAddChecker([PointerClass(_type), PointerClass(_type2)], [PointerClass(_type3)]),
                RQAddTemplate.referenceTemplate, GAP9Transformer)
    for _type in [int8_t, uint8_t]
    for _type2 in [int8_t, uint8_t]
    for _type3 in [int8_t, uint8_t]
]

GAP9AddBindings = [
    NodeBinding(AddChecker([PointerClass(type1), PointerClass(type2)], [PointerClass(int32_t)]),
                AddTemplate.referenceTemplate, GAP9Transformer)
    for type1 in IntegerDataTypes
    for type2 in IntegerDataTypes
] + [
    NodeBinding(AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAddTemplate.referenceTemplate, GAP9Transformer)
]

GAP9RQSConv2DBindings = [
    NodeBinding(
        PULPConvChecker([
            PointerClass(type1),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(type2)]), ConvTemplate.PULPConv2D_8_Template, GAP9Transformer)
    for type1, type2 in zip([int8_t, int8_t, uint8_t, uint8_t], [int8_t, uint8_t, int8_t, uint8_t])
]

GAP9RQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker([
            PointerClass(type1),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(type2)]), ConvTemplate.PULPDWConv2D_8_Template, GAP9Transformer)
    for type1, type2 in zip([int8_t, int8_t, uint8_t, uint8_t], [int8_t, uint8_t, int8_t, uint8_t])
]

GAP9RQSGEMM_8_Binding = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), GEMMTemplate.PULPGEMM_8_Template,
        GAP9Transformer) for type1, type2 in zip([int8_t, uint8_t, int8_t, uint8_t], [int8_t, uint8_t, uint8_t, int8_t])
]

GAP9FloatGEMMBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGemmTemplate.referenceTemplate,
        GAP9Transformer)
]

GAP9FloatConv2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatConvTemplate.reference2DIm2ColTemplate,
        GAP9Transformer)
]

GAP9FloatDWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(float_type), PointerClass(float_type),
             PointerClass(float_type)], [PointerClass(float_type)]), FloatConvTemplate.referenceDW2DIm2ColTemplate,
        GAP9Transformer) for float_type in FloatDataTypes
]

GAP9RQSMatrixVecBindings = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), MatrixVectorTemplate.referenceTemplate,
        GAP9Transformer) for type1, type2 in zip([int8_t], [int8_t])
]

GAP9RQSTallGEMMBindings = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), TallGEMMTemplate.referenceTemplate,
        GAP9Transformer) for type1, type2 in zip([int8_t], [int8_t])
]

GAP9RQSGEMMBindings = GAP9RQSGEMM_8_Binding

GAP9MaxPool2DBindings = [
    NodeBinding(PULPMaxPoolChecker([PointerClass(type)], [PointerClass(type)]),
                MaxPoolTemplate.PULPMaxPool2D_8_Template, GAP9Transformer) for type in [int8_t, uint8_t]
] + [
    NodeBinding(PULPMaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMaxPoolTemplate.referenceTemplate, GAP9Transformer)
]

GAP9Conv1DBinding = NodeBinding(
    PULPConvChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), ConvTemplate.PULPConv1D_8_Template, GAP9Transformer)

GAP9DWConv1DBinding = NodeBinding(
    PULPConvChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), ConvTemplate.PULPDWConv1D_8_Template, GAP9Transformer)

GAP9MatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                GEMMTemplate.PULPMM_8_Template, GAP9ClusterTransformer)
] + [
    NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMatMulTemplate.referenceTemplate, GAP9Transformer)
]

GAP9ReduceMeanBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(type)], [PointerClass(type)]), ReduceMeanTemplate.referenceTemplate,
                GAP9ClusterTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(ReduceMeanChecker([PointerClass(float_type), PointerClass(integer_type)], [PointerClass(float_type)]),
                FloatReduceMeanTemplate.referenceTemplate, GAP9ClusterTransformer)
    for integer_type in SignedIntegerDataTypes
    for float_type in FloatDataTypes
]

GAP9ReduceSumBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatReduceSumTemplate.referenceTemplate, GAP9ClusterTransformer)
]

GAP9UniformRQSBindings = [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(int8_t)]),
        UniformRequantShiftTemplate.referenceTemplate, GAP9Transformer) for type in IntegerDataTypes
]

GAP9RQSBindings = [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(int8_t)]),
        RequantShiftTemplate.referenceTemplate, GAP9Transformer) for type in IntegerDataTypes
] + [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(uint8_t)]),
        RequantShiftTemplate.referenceTemplate, GAP9Transformer) for type in IntegerDataTypes
]

GAP9SoftmaxBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(_type)], [PointerClass(uint8_t)]), iSoftmaxTemplate.referenceTemplate,
                GAP9Transformer) for _type in [int8_t, uint8_t]
] + [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSoftmaxTemplate.referenceTemplate, GAP9Transformer)
]

GAP9SoftmaxGradBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSoftmaxTemplate.referenceGradientTemplate, GAP9Transformer)
]

GAP9SoftmaxCrossEntropyLossBindings = [
    NodeBinding(
        SoftmaxCrossEntropyLossChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
        SoftmaxCrossEntropyLossTemplate.referenceTemplate, GAP9Transformer) for type in IntegerDataTypes
]

GAP9SoftmaxCrossEntropyLossGradBindings = [
    NodeBinding(
        SoftmaxCrossEntropyLossChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
        SoftmaxCrossEntropyLossTemplate.referenceGradientTemplate, GAP9Transformer) for type in IntegerDataTypes
]

GAP9SGDBindings = [
    NodeBinding(SGDChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                SGDTemplate.referenceTemplate, GAP9Transformer)
]

GAP9TransposeBindings = [
    NodeBinding(TransposeChecker([PointerClass(type)], [PointerClass(type)]), TransposeTemplate.referenceTemplate,
                GAP9Transformer) for type in IntegerDataTypes
] + [
    NodeBinding(TransposeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                TransposeTemplate.referenceTemplate, GAP9Transformer)
]

GAP9ConcatBindings = [
    NodeBinding(ConcatChecker([PointerClass(type), PointerClass(type)], [PointerClass(type)]),
                ConcatTemplate.referenceTemplate, GAP9ClusterTransformer) for type in IntegerDataTypes
]

GAP9iRMSNormBindings = [
    NodeBinding(LayerNormChecker([PointerClass(int8_t), PointerClass(int32_t)], [PointerClass(int8_t)]),
                iRMSNormTemplate.referenceTemplate, GAP9Transformer)
]

GAP9iHardswishBindings = [
    NodeBinding(HardswishChecker([PointerClass(int8_t)], [PointerClass(int32_t)]), iHardswishTemplate.referenceTemplate,
                GAP9ClusterTransformer)
]

GAP9RQSiHardswishBindings = [
    NodeBinding(
        RQHardswishChecker([PointerClass(int8_t),
                            PointerClass(int32_t),
                            PointerClass(int32_t),
                            PointerClass(int32_t)], [PointerClass(int8_t)]), RQSiHardswishTemplate.referenceTemplate,
        GAP9Transformer)
]

GAP9iRQSGELUBindings = [
    NodeBinding(
        GELUChecker([PointerClass(int8_t),
                     PointerClass(int32_t),
                     PointerClass(int32_t),
                     PointerClass(int32_t)], [PointerClass(int8_t)]), RQSiGELUTemplate.referenceTemplate,
        GAP9ClusterTransformer)
]

GAP9MulBindings = [
    NodeBinding(MulChecker([PointerClass(typeA), PointerClass(typeB)], [PointerClass(int32_t)]),
                MulTemplate.referenceTemplate, GAP9Transformer)
    for typeA, typeB in itertools.product(SignedIntegerDataTypes, SignedIntegerDataTypes)
] + [
    NodeBinding(MulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMulTemplate.referenceTemplate, GAP9Transformer)
]

GAP9ReluBinding = NodeBinding(ReluChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                              FloatReluTemplate.referenceTemplate, GAP9Transformer)

GAP9LayernormBinding = NodeBinding(
    LayerNormChecker(
        [PointerClass(float32_t), PointerClass(float32_t),
         PointerClass(float32_t)], [PointerClass(float32_t)]), FloatLayernormTemplate.referenceTemplate,
    GAP9Transformer)

GAP9FloatGELUBinding = NodeBinding(
    GELUChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    FloatGELUTemplate.referenceTemplate, GAP9Transformer)

GAP9GatherBindings = [
    NodeBinding(GatherChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
                GatherTemplate.referenceTemplate, GAP9Transformer) for type in IntegerDataTypes
]

GAP9QuantBindings = [
    NodeBinding(QuantChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), QuantTemplate.referenceTemplate,
                GAP9Transformer),
]

GAP9DequantBindings = [
    NodeBinding(DequantChecker([PointerClass(int8_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                GAP9Transformer),
] + [
    NodeBinding(DequantChecker([PointerClass(int32_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                GAP9Transformer),
]
