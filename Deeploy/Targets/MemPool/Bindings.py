# ----------------------------------------------------------------------
#
# File: MemPoolBindings.py
#
# Last edited: 13.11.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Philip Wiese, ETH Zurich
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
