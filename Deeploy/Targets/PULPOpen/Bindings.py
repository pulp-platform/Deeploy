# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import itertools
from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration, MemoryPassthroughGeneration
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, SignedIntegerDataTypes, float32_t, \
    int8_t, int32_t, int64_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding, NodeTemplate
from Deeploy.FutureExtension.Bindings.AutoFutureBinding import AutoFutureBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import AddTemplate, ConcatTemplate, DequantTemplate, FloatReduceSumTemplate, \
    GatherTemplate, QuantTemplate, RQSiGELUTemplate, SliceTemplate, iHardswishTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, BatchNormInternalChecker, \
    BatchNormalizationGradChecker, ConcatChecker, ConvChecker, DequantChecker, \
    GatherChecker, GELUChecker, GEMMChecker, GlobalAveragePoolChecker, GlobalAveragePoolGradChecker, \
    HardswishChecker, InPlaceAccumulatorV2Checker, LayerNormChecker, MatMulChecker, MaxPoolGradChecker, MulChecker, \
    MSELossChecker, QuantChecker, ReduceMeanChecker, ReluChecker, ReshapeChecker, RQAddChecker, RQHardswishChecker, \
    SGDChecker, SliceChecker, SoftmaxChecker, SoftmaxCrossEntropyLossChecker, TransposeChecker, \
    PULPConvGradBChecker
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterSynch import PULPSynchCoresPass
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPClusterTiling import PULPClusterTiling
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPL3Tiling import PULPL3Tiling
from Deeploy.Targets.PULPOpen.CodeTransformationPasses.PULPProfileUntiled import PULPProfileUntiled
from Deeploy.Targets.PULPOpen.DataTypes import PULPDMAFuture
from Deeploy.Targets.PULPOpen.DMA.L3Dma import l3DmaHack
from Deeploy.Targets.PULPOpen.DMA.MchanDma import MchanDma
from Deeploy.Targets.PULPOpen.Templates import ConvTemplate, DMASliceTemplate, FloatAddTemplate, FloatAveragePoolTemplate, \
    FloatBatchNormTemplate, FloatConvGradTemplate, FloatConvTemplate, FloatGELUTemplate, FloatGemmTemplate, \
    FloatGlobalAveragePoolTemplate, \
    FloatInPlaceAccumulatorV2Template, FloatLayernormTemplate, FloatMatMulTemplate, \
    FloatMaxPoolTemplate, FloatMulTemplate, FloatReduceMeanTemplate, FloatReluTemplate, FloatSoftmaxTemplate, \
    GEMMTemplate, MatrixVectorTemplate, MaxPool2DTemplate, MSELossTemplate, MulTemplate, ReduceMeanTemplate, \
    RequantShiftTemplate, ReshapeTemplate, RQAddTemplate, RQSiHardswishTemplate, SGDTemplate, \
    SoftmaxCrossEntropyLossTemplate, \
    TallGEMMTemplate, TransposeTemplate, UniformRequantShiftTemplate, iRMSNormTemplate, iSoftmaxTemplate
from Deeploy.Targets.PULPOpen.TypeCheckers import PULPConvChecker, PULPLinearChecker, \
    PULPMaxPoolChecker, PULPRequantShiftChecker
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement, \
    TilingVariableReplacementUpdate

_clusterEntryClosureCallTemplate = NodeTemplate("""
// ${closureName} CLOSURE CALL
static struct pi_cluster_task cluster_task;

pi_cluster_task(&cluster_task, ${closureName}, &${closureStructArgName});
cluster_task.stack_size = 5000;
cluster_task.slave_stack_size = 3800;
pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
//pi_cluster_close(&cluster_dev);
""")

_clusterForkClosureCallTemplate = NodeTemplate("""
pi_cl_team_fork(NUM_CORES, (void*)${closureName}, &${closureStructArgName});
""")

SkipTransformer = CodeTransformation(
    [ArgumentStructGeneration(),
     MemoryPassthroughGeneration("L.*"),
     MemoryPassthroughGeneration(),
     FutureGeneration()])

FunctionCallClosure = partial(ClosureGeneration, closureSuffix = "_closure")
ClusterClosure = partial(ClosureGeneration,
                         closureSuffix = "_cluster_entry",
                         closureCallTemplate = _clusterEntryClosureCallTemplate)
ForkClosure = partial(ClosureGeneration,
                      closureSuffix = "_cluster_fork",
                      closureCallTemplate = _clusterForkClosureCallTemplate)

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")

MemoryAwareClusterClosure = partial(MemoryAwareClosureGeneration,
                                    closureSuffix = "_cluster_entry",
                                    closureCallTemplate = _clusterEntryClosureCallTemplate,
                                    startRegion = "L2",
                                    endRegion = "L1")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration,
                                         closureSuffix = "_closure",
                                         startRegion = "L2",
                                         endRegion = "L1")

L3MemoryAwareFunctionCallClosure = partial(ClosureGeneration, closureSuffix = "_closure_L3")

MemoryAwareForkTransformer = CodeTransformation([
    ArgumentStructGeneration(),
    ForkClosure(generateStruct = False),
    FutureGeneration(),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    FunctionCallClosure(writeback = True),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration()
])

ForkTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    PULPSynchCoresPass(),
    ForkClosure(writeback = False, generateStruct = True),
    TilingVariableReplacementUpdate("L1"),
    PULPClusterTiling("L2", "L1", MchanDma()),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    TilingVariableReplacement("L2"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    PULPL3Tiling("L3", "L2", l3DmaHack),
    PULPProfileUntiled(),
    ArgumentStructGeneration(),
    L3MemoryAwareFunctionCallClosure(writeback = False),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

ClusterTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False, generateStruct = True),
    TilingVariableReplacementUpdate("L1"),
    PULPClusterTiling("L2", "L1", MchanDma()),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    TilingVariableReplacement("L2"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    PULPL3Tiling("L3", "L2", l3DmaHack),
    PULPProfileUntiled(),
    ArgumentStructGeneration(),
    L3MemoryAwareFunctionCallClosure(writeback = False),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

SimpleTransformer = CodeTransformation([
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration("L3.*"),
    MemoryManagementGeneration(),
])

PULPDMASliceBindings = [
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

PULPSliceBindings = [
    NodeBinding(
        SliceChecker(
            [
                PointerClass(float_type),  # data_in
                PointerClass(int_type),  # starts
                PointerClass(int_type),  # ends
                PointerClass(int_type),  # axes
                PointerClass(int_type)  # steps
            ],
            [PointerClass(float_type)]),
        SliceTemplate.referenceTemplate,
        ForkTransformer) for float_type in FloatDataTypes for int_type in IntegerDataTypes
]

PULPReshapeBindings = [
    NodeBinding(ReshapeChecker([PointerClass(type), PointerClass(int64_t)], [PointerClass(type)]),
                ReshapeTemplate.referenceTemplate, SkipTransformer) for type in IntegerDataTypes + FloatDataTypes
]

PULPRQAddBindings = [
    NodeBinding(RQAddChecker([PointerClass(_type), PointerClass(_type2)], [PointerClass(_type3)]),
                RQAddTemplate.referenceTemplate, ForkTransformer)
    for _type in [int8_t, uint8_t]
    for _type2 in [int8_t, uint8_t]
    for _type3 in [int8_t, uint8_t]
]

PULPAddBindings = [
    NodeBinding(AddChecker([PointerClass(type1), PointerClass(type2)], [PointerClass(int32_t)]),
                AddTemplate.referenceTemplate, ForkTransformer)
    for type1 in IntegerDataTypes
    for type2 in IntegerDataTypes
] + [
    NodeBinding(AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAddTemplate.referenceTemplate, ForkTransformer)
] + [
    NodeBinding(
        AddChecker([PointerClass(float32_t), PointerClass(float32_t),
                    PointerClass(float32_t)], [PointerClass(float32_t)]), FloatAddTemplate.referenceTemplate,
        ForkTransformer)
]

PULPRQSConv2DBindings = [
    NodeBinding(
        PULPConvChecker([
            PointerClass(type1),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(type2)]), ConvTemplate.PULPConv2D_8_Template, ForkTransformer)
    for type1, type2 in zip([int8_t, int8_t, uint8_t, uint8_t], [int8_t, uint8_t, int8_t, uint8_t])
]

PULPRQSDWConv2DBindings = [
    NodeBinding(
        PULPConvChecker([
            PointerClass(type1),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(type2)]), ConvTemplate.PULPDWConv2D_8_Template, ForkTransformer)
    for type1, type2 in zip([int8_t, int8_t, uint8_t, uint8_t], [int8_t, uint8_t, int8_t, uint8_t])
]

PULPRQSGEMM_8_Binding = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), GEMMTemplate.PULPGEMM_8_Template,
        ForkTransformer) for type1, type2 in zip([int8_t, uint8_t, int8_t, uint8_t], [int8_t, uint8_t, uint8_t, int8_t])
]

PULPFloatGEMMBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGemmTemplate.referenceTemplate,
        ForkTransformer)
]

PULPFloatConv2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatConvTemplate.reference2DIm2ColTemplate,
        ForkTransformer)
]

PULPFloatConvGradW2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t)],
                    [PointerClass(float32_t)]), FloatConvGradTemplate.referenceConvGradW2DIm2ColTemplate,
        ClusterTransformer)
]

PULPFloatConvGradX2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t)],
                    [PointerClass(float32_t)]), FloatConvGradTemplate.referenceConvGradX2DIm2ColTiledTemplate,
        ForkTransformer)
]

PULPFloatDWConv2DBindings = [
    NodeBinding(
        ConvChecker(
            [PointerClass(float_type), PointerClass(float_type),
             PointerClass(float_type)], [PointerClass(float_type)]), FloatConvTemplate.referenceDW2DIm2ColTemplate,
        ForkTransformer) for float_type in FloatDataTypes
]

PULPFloatDWConvGradX2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t)],
                    [PointerClass(float32_t)]), FloatConvGradTemplate.referenceDWConvGradX2DTiledTemplate,
        ForkTransformer)
]

PULPFloatDWConvGradW2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t)],
                    [PointerClass(float32_t)]), FloatConvGradTemplate.referenceDWConvGradW2DTemplate,
        ClusterTransformer)
]

PULPFloatPWConvGradW2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t)],
                    [PointerClass(float32_t)]), FloatConvGradTemplate.referencePWConvGradW2DTemplate,
        ClusterTransformer)
]

PULPFloatPWConvGradX2DBindings = [
    NodeBinding(
        ConvChecker([PointerClass(float32_t), PointerClass(float32_t)],
                    [PointerClass(float32_t)]), FloatConvGradTemplate.referencePWConvGradX2DTemplate,
        ClusterTransformer)
]

PULPFloatConvGradBBindings = [
    NodeBinding(
        PULPConvGradBChecker([PointerClass(float32_t)],
                             [PointerClass(float32_t)]), FloatConvGradTemplate.referenceConvGradB2DTemplate,
        ClusterTransformer)
]

# 5 inputs (X, gamma, beta, running_mean, running_var), 5 outputs (Y, urm, urv, saved_mean, saved_inv_std)
PULPBatchNormInternalBindings = [
    NodeBinding(
        BatchNormInternalChecker(
            [PointerClass(float32_t)] * 5,
            [PointerClass(float32_t)] * 5), FloatBatchNormTemplate.batchNormInternalTemplate,
        ForkTransformer)
]

# 5 inputs (dY, X, gamma, saved_mean, saved_inv_std), 3 outputs (dX, dgamma, dbeta)
PULPBatchNormalizationGradBindings = [
    NodeBinding(
        BatchNormalizationGradChecker(
            [PointerClass(float32_t)] * 5,
            [PointerClass(float32_t)] * 3), FloatBatchNormTemplate.batchNormGradTemplate,
        ForkTransformer)
]


PULPRQSMatrixVecBindings = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), MatrixVectorTemplate.referenceTemplate,
        ForkTransformer) for type1, type2 in zip([int8_t], [int8_t])
]

PULPRQSTallGEMMBindings = [
    NodeBinding(
        PULPLinearChecker([PointerClass(type1),
                           PointerClass(int8_t),
                           PointerClass(int32_t),
                           PointerClass(int32_t)], [PointerClass(type2)]), TallGEMMTemplate.referenceTemplate,
        ForkTransformer) for type1, type2 in zip([int8_t], [int8_t])
]

PULPRQSGEMMBindings = PULPRQSGEMM_8_Binding

PULPMaxPool2DBindings = [
    NodeBinding(PULPMaxPoolChecker([PointerClass(type)], [PointerClass(type)]),
                MaxPool2DTemplate.PULPMaxPool2D_8_Template, ForkTransformer) for type in [int8_t, uint8_t]
] + [
    NodeBinding(PULPMaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMaxPoolTemplate.referenceTemplate, ForkTransformer)
]

PULPAveragePool2DBindings = [
    NodeBinding(PULPMaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAveragePoolTemplate.referenceTemplate, ForkTransformer)
]

PULPAveragePoolGrad2DBindings = [
    NodeBinding(PULPMaxPoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAveragePoolTemplate.referenceGradTemplate, ForkTransformer)
]

# 1 input (data_in [N,C,H,W]), 1 output (data_out [N,C,1,1])
PULPGlobalAveragePool2DBindings = [
    NodeBinding(
        GlobalAveragePoolChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
        FloatGlobalAveragePoolTemplate.globalAveragePoolTemplate,
        ForkTransformer)
]

# 1 input (dY [N,C,1,1]), 1 output (dX [N,C,H,W])
PULPGlobalAveragePoolGrad2DBindings = [
    NodeBinding(
        GlobalAveragePoolGradChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
        FloatGlobalAveragePoolTemplate.globalAveragePoolGradTemplate,
        ForkTransformer)
]

PULPMaxPoolGrad2DBindings = [
    NodeBinding(
        MaxPoolGradChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        FloatMaxPoolTemplate.referenceGradTemplate, ForkTransformer)
]


PULPConv1DBinding = NodeBinding(
    PULPConvChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), ConvTemplate.PULPConv1D_8_Template, ForkTransformer)

PULPDWConv1DBinding = NodeBinding(
    PULPConvChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), ConvTemplate.PULPDWConv1D_8_Template, ForkTransformer)

PULPMatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                GEMMTemplate.PULPMM_8_Template, ClusterTransformer)
] + [
    NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMatMulTemplate.referenceTemplate, ForkTransformer)
]

PULPReduceMeanBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(type)], [PointerClass(type)]), ReduceMeanTemplate.referenceTemplate,
                ClusterTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(ReduceMeanChecker([PointerClass(float_type), PointerClass(integer_type)], [PointerClass(float_type)]),
                FloatReduceMeanTemplate.referenceTemplate, ForkTransformer)
    for integer_type in SignedIntegerDataTypes
    for float_type in FloatDataTypes
]

PULPReduceSumBindings = [
    NodeBinding(ReduceMeanChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatReduceSumTemplate.referenceTemplate, ClusterTransformer)
]

PULPUniformRQSBindings = [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(int8_t)]),
        UniformRequantShiftTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPRQSBindings = [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(int8_t)]),
        RequantShiftTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(
        PULPRequantShiftChecker([PointerClass(type), PointerClass(int32_t),
                                 PointerClass(int32_t)], [PointerClass(uint8_t)]),
        RequantShiftTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPSoftmaxBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(_type)], [PointerClass(uint8_t)]), iSoftmaxTemplate.referenceTemplate,
                ForkTransformer) for _type in [int8_t, uint8_t]
] + [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSoftmaxTemplate.referenceTemplate, ForkTransformer)
]

PULPSoftmaxGradBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatSoftmaxTemplate.referenceGradientTemplate, ForkTransformer)
]

PULPSoftmaxCrossEntropyLossBindings = [
    NodeBinding(
        SoftmaxCrossEntropyLossChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
        SoftmaxCrossEntropyLossTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

# Dual-output binding: outputs[0]=loss (scalar), outputs[1]=log_prob
PULPSoftmaxCrossEntropyLossDualOutputBindings = [
    NodeBinding(
        SoftmaxCrossEntropyLossChecker([PointerClass(float32_t), PointerClass(type)],
                                       [PointerClass(float32_t), PointerClass(float32_t)]),
        SoftmaxCrossEntropyLossTemplate.referenceDualOutputTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPSoftmaxCrossEntropyLossGradBindings = [
    NodeBinding(
        SoftmaxCrossEntropyLossChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
        SoftmaxCrossEntropyLossTemplate.referenceGradientTemplate, ForkTransformer) for type in IntegerDataTypes
]

PULPMSELossBindings = [
    NodeBinding(MSELossChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                MSELossTemplate.referenceTemplate, ForkTransformer)
]

PULPMSELossGradBindings = [
    NodeBinding(MSELossChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                MSELossTemplate.referenceGradientTemplate, ForkTransformer)
]

PULPSGDBindings = [
    NodeBinding(SGDChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                SGDTemplate.referenceTemplate, ForkTransformer)
]

PULPInPlaceAccumulatorV2Bindings = [
    NodeBinding(
        InPlaceAccumulatorV2Checker(
            [PointerClass(float32_t), PointerClass(float32_t), PointerClass(uint8_t)], [PointerClass(float32_t)]),
        FloatInPlaceAccumulatorV2Template.referenceTemplate, ForkTransformer)
]

PULPInPlaceAccumulatorV2TiledBindings = [
    NodeBinding(
        InPlaceAccumulatorV2Checker(
            [PointerClass(float32_t), PointerClass(float32_t), PointerClass(uint8_t)], [PointerClass(float32_t)]),
        FloatInPlaceAccumulatorV2Template.tiledReferenceTemplate, ForkTransformer)
]

PULPTransposeBindings = [
    NodeBinding(TransposeChecker([PointerClass(type)], [PointerClass(type)]), TransposeTemplate.referenceTemplate,
                ForkTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(TransposeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                TransposeTemplate.referenceTemplate, ForkTransformer)
]

PULPConcatBindings = [
    NodeBinding(ConcatChecker([PointerClass(type), PointerClass(type)], [PointerClass(type)]),
                ConcatTemplate.referenceTemplate, ClusterTransformer) for type in IntegerDataTypes
] + [
    NodeBinding(ConcatChecker([PointerClass(float_type), PointerClass(float_type)], [PointerClass(float_type)]),
                ConcatTemplate.referenceTemplate, ClusterTransformer) for float_type in FloatDataTypes
]

PULPiRMSNormBindings = [
    NodeBinding(LayerNormChecker([PointerClass(int8_t), PointerClass(int32_t)], [PointerClass(int8_t)]),
                iRMSNormTemplate.referenceTemplate, ForkTransformer)
]

PULPiHardswishBindings = [
    NodeBinding(HardswishChecker([PointerClass(int8_t)], [PointerClass(int32_t)]), iHardswishTemplate.referenceTemplate,
                ClusterTransformer)
]
PULPRQSiHardswishBindings = [
    NodeBinding(
        RQHardswishChecker([PointerClass(int8_t),
                            PointerClass(int32_t),
                            PointerClass(int32_t),
                            PointerClass(int32_t)], [PointerClass(int8_t)]), RQSiHardswishTemplate.referenceTemplate,
        ForkTransformer)
]

PULPiRQSGELUBindings = [
    NodeBinding(
        GELUChecker([PointerClass(int8_t),
                     PointerClass(int32_t),
                     PointerClass(int32_t),
                     PointerClass(int32_t)], [PointerClass(int8_t)]), RQSiGELUTemplate.referenceTemplate,
        ClusterTransformer)
]

PULPMulBindings = [
    NodeBinding(MulChecker([PointerClass(typeA), PointerClass(typeB)], [PointerClass(int32_t)]),
                MulTemplate.referenceTemplate, ForkTransformer)
    for typeA, typeB in itertools.product(SignedIntegerDataTypes, SignedIntegerDataTypes)
] + [
    NodeBinding(MulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMulTemplate.referenceTemplate, ForkTransformer)
]

PULPReluBinding = NodeBinding(ReluChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                              FloatReluTemplate.referenceTemplate, ForkTransformer)

PULPReluGradBinding = NodeBinding(
    ReluChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    FloatReluTemplate.referenceGradTemplate, ForkTransformer)

PULPLayernormBinding = NodeBinding(
    LayerNormChecker(
        # inputs: data_in (X), weight (scale/gamma), bias (beta)
        [PointerClass(float32_t), PointerClass(float32_t),
         PointerClass(float32_t)],
        # outputs: data_out (Y), mean stash, inv_std_dev stash
        [PointerClass(float32_t), PointerClass(float32_t),
         PointerClass(float32_t)]), FloatLayernormTemplate.referenceTemplate,
    ForkTransformer)

PULPLayernormGradBinding = NodeBinding(
    LayerNormChecker(
        # inputs: grad_in (dY), data_in (X), weight (scale/gamma),
        #         mean stash, inv_std_dev stash
        [PointerClass(float32_t),
         PointerClass(float32_t),
         PointerClass(float32_t),
         PointerClass(float32_t),
         PointerClass(float32_t)],
        # outputs: grad_out (dX), weight_grad (dscale), bias_grad (dbias)
        [PointerClass(float32_t),
         PointerClass(float32_t),
         PointerClass(float32_t)]), FloatLayernormTemplate.referenceGradTemplate,
    ForkTransformer)

PULPFloatGELUBinding = NodeBinding(
    GELUChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    FloatGELUTemplate.referenceTemplate, ForkTransformer)

PULPFloatGELUGradBinding = NodeBinding(
    GELUChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
    FloatGELUTemplate.referenceGradTemplate, ForkTransformer)

PULPGatherBindings = [
    NodeBinding(GatherChecker([PointerClass(float32_t), PointerClass(type)], [PointerClass(float32_t)]),
                GatherTemplate.referenceTemplate, ForkTransformer) for type in IntegerDataTypes
]

BasicQuantBindings = [
    NodeBinding(QuantChecker([PointerClass(float32_t)], [PointerClass(int8_t)]), QuantTemplate.referenceTemplate,
                ForkTransformer),
]

BasicDequantBindings = [
    NodeBinding(DequantChecker([PointerClass(int8_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                ForkTransformer),
] + [
    NodeBinding(DequantChecker([PointerClass(int32_t)], [PointerClass(float32_t)]), DequantTemplate.referenceTemplate,
                ForkTransformer),
]

