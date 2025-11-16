# SPDX-FileCopyrightText: 2022 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes, int8_t, int32_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.TypeCheckers import ConvChecker, GEMMChecker, MatMulChecker, MaxPoolChecker, MHSAChecker, \
    RequantShiftChecker, RQGEMMChecker, RQMatMulChecker, SoftmaxChecker
from Deeploy.Targets.MemPool.Templates import ConvTemplate, DWConvTemplate, GemmTemplate, ITAMaxTemplate, ITATemplate, \
    MatMulTemplate, MaxPoolTemplate, RequantShiftTemplate, RQGemmTemplate, RQMatMulTemplate

BasicTransformer = CodeTransformation([MemoryManagementGeneration(), ArgumentStructGeneration(), FutureGeneration()])

MemPoolConv1D_8_8_32_Binding = NodeBinding(
    ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
    ConvTemplate.MemPoolParallel1DTemplate, BasicTransformer)
MemPoolConv2D_8_8_32_Binding = NodeBinding(
    ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
    ConvTemplate.MemPoolParallel2DTemplate, BasicTransformer)
MemPoolDWConv1D_8_8_32_Binding = NodeBinding(
    ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
    DWConvTemplate.MemPoolParallel1DTemplate, BasicTransformer)
MemPoolDWConv2D_8_8_32_Binding = NodeBinding(
    ConvChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
    DWConvTemplate.MemPoolParallel2DTemplate, BasicTransformer)
MemPoolGEMMBinding_8_8_32_32 = NodeBinding(
    GEMMChecker(
        [PointerClass(int8_t), PointerClass(int8_t), PointerClass(int32_t)], [PointerClass(int32_t)]),
    GemmTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolITASoftmaxBinding_8_8 = NodeBinding(SoftmaxChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
                                           ITAMaxTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolMatMul_8_8_32_Binding = NodeBinding(
    MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
    MatMulTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolMaxPool2D_8_8_Binding = NodeBinding(MaxPoolChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
                                           MaxPoolTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolMHSA_1H_INT8_Binding = NodeBinding(
    MHSAChecker(
        [PointerClass(int8_t), PointerClass(int8_t), PointerClass(int8_t)] +
        [PointerClass(int8_t), PointerClass(int32_t)] * 4, [PointerClass(int8_t)]),
    ITATemplate.MemPoolParallelTemplate_1H, BasicTransformer)
MemPoolMHSA_2H_INT8_Binding = NodeBinding(
    MHSAChecker(
        [PointerClass(int8_t), PointerClass(int8_t), PointerClass(int8_t)] +
        [PointerClass(int8_t), PointerClass(int32_t)] * 4, [PointerClass(int8_t)]),
    ITATemplate.MemPoolParallelTemplate_2H, BasicTransformer)
MemPoolMHSA_4H_INT8_Binding = NodeBinding(
    MHSAChecker(
        [PointerClass(int8_t), PointerClass(int8_t), PointerClass(int8_t)] +
        [PointerClass(int8_t), PointerClass(int32_t)] * 4, [PointerClass(int8_t)]),
    ITATemplate.MemPoolParallelTemplate_4H, BasicTransformer)
MemPoolRQGEMMBinding_8_8_32_32_32_8 = NodeBinding(
    RQGEMMChecker([
        PointerClass(int8_t),
        PointerClass(int8_t),
        PointerClass(int32_t),
        PointerClass(int32_t),
        PointerClass(int32_t)
    ], [PointerClass(int8_t)]), RQGemmTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolRQMatMul_8_8_32_32_Binding = NodeBinding(
    RQMatMulChecker(
        [PointerClass(int8_t), PointerClass(int8_t),
         PointerClass(int32_t),
         PointerClass(int32_t)], [PointerClass(int8_t)]), RQMatMulTemplate.MemPoolParallelTemplate, BasicTransformer)
MemPoolRQSBindings_x_32_32_8 = [
    NodeBinding(
        RequantShiftChecker([PointerClass(_type), PointerClass(int32_t),
                             PointerClass(int32_t)], [PointerClass(int8_t)]),
        RequantShiftTemplate.MemPoolParallelTemplate, BasicTransformer) for _type in IntegerDataTypes
]
