# ----------------------------------------------------------------------
#
# File: MemPoolPlatform.py
#
# Last edited: 17.12.2022
#
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author:
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

from typing import Dict

import numpy as np

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicConv1DBinding, BasicConv2DBinding, \
    BasicDebugPrintBindings, BasicDWConv1DBinding, BasicDWConv2DBinding, BasicGatherBindings, BasicGELUBinding, \
    BasicIntegerDivBinding, BasicLayerNormBinding, BasicMulBindings, BasicPad1DBindings, BasicPad2DBindings, \
    BasicReduceMeanBindings, BasicReduceSumBindings, BasicReshapeBindings, BasicRQIntegerDivBinding, \
    BasicRQSGELUBinding, BasicSliceBindings, BasicSoftmaxBinding, BasicTransposeBindings, DummyBinding
from Deeploy.Targets.Generic.Layers import AddLayer, ConvLayer, DebugPrintLayer, GatherLayer, GEMMLayer, \
    IntegerDivLayer, ITAMaxLayer, MatMulLayer, MaxPoolLayer, MHSALayer, MulLayer, PadLayer, ReduceMeanLayer, \
    ReduceSumLayer, RequantShiftLayer, ReshapeLayer, RQGEMMLayer, RQIntegerDivLayer, RQMatMulLayer, RQSiGELULayer, \
    SliceLayer, TransposeLayer, iGELULayer, iLayerNormLayer, iSoftmaxLayer
from Deeploy.Targets.Generic.Parsers import AddParser, DebugParser, DummyParser, FlattenParser, GatherParser, \
    GenericConv1DParser, GenericConv2DParser, GenericDWConv1DParser, GenericDWConv2DParser, GenericGEMMParser, \
    GenericMaxPool2DParser, IntegerDivParser, ITAMaxParser, MatMulParser, MulParser, Pad1DParser, Pad2DParser, \
    ReduceMeanParser, ReduceSumParser, RequantShiftParser, ReshapeParser, RQGEMMParser, RQIntegerDivParser, \
    RQMatMulParser, RQSiGELUParser, SliceParser, TransposeParser, UnsqueezeParser, iGELUParser, iLayerNormParser, \
    iSoftmaxParser
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ExtractPaddingFromConvPass, \
    ExtractPaddingFromPoolPass, MatMulAddMergePass, MergeConstAddAndRequantPass, SplitAddPass, iGELURequantMergePass
from Deeploy.Targets.MemPool.Bindings import MemPoolConv1D_8_8_32_Binding, MemPoolConv2D_8_8_32_Binding, \
    MemPoolDWConv1D_8_8_32_Binding, MemPoolDWConv2D_8_8_32_Binding, MemPoolGEMMBinding_8_8_32_32, \
    MemPoolITASoftmaxBinding_8_8, MemPoolMatMul_8_8_32_Binding, MemPoolMaxPool2D_8_8_Binding, \
    MemPoolMHSA_1H_INT8_Binding, MemPoolMHSA_2H_INT8_Binding, MemPoolMHSA_4H_INT8_Binding, \
    MemPoolRQGEMMBinding_8_8_32_32_32_8, MemPoolRQMatMul_8_8_32_32_Binding, MemPoolRQSBindings_x_32_32_8
from Deeploy.Targets.MemPool.Parsers import MemPoolITAM4HSAParser, MemPoolM1HSAParser, MemPoolM2HSAParser
from Deeploy.Targets.MemPool.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.MemPool.TopologyOptimizationPasses.Passes import MemPoolFuseMHSAPass, \
    MemPoolGEMMRequantMergePass, MemPoolMatMulRequantMergePass, MemPoolSplitMHSAPass

# Fallback bindings from the generic platform
# (they support a wider range of attribute values)
GenericConv1D_Mapper = NodeMapper(GenericConv1DParser(), [BasicConv1DBinding])
GenericDWConv1D_Mapper = NodeMapper(GenericDWConv1DParser(), [BasicDWConv1DBinding])
GenericConv2D_Mapper = NodeMapper(GenericConv2DParser(), [BasicConv2DBinding])
GenericDWConv2D_Mapper = NodeMapper(GenericDWConv2DParser(), [BasicDWConv2DBinding])

GenericConv_Mappers = [GenericConv2D_Mapper, GenericDWConv2D_Mapper, GenericConv1D_Mapper, GenericDWConv1D_Mapper]

# Basic bindings
Add_Mapper = NodeMapper(AddParser(), BasicAddBindings)
DebugPrint_Mapper = NodeMapper(DebugParser(), BasicDebugPrintBindings)
Flatten_Mapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
Gather_Mapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELU_Mapper = NodeMapper(iGELUParser(), [BasicGELUBinding])
iLayerNorm_Mapper = NodeMapper(iLayerNormParser(), [BasicLayerNormBinding])
IntegerDiv_Mapper = NodeMapper(IntegerDivParser(), [BasicIntegerDivBinding])
ITAMaxMapper = NodeMapper(ITAMaxParser(), [MemPoolITASoftmaxBinding_8_8])
Mul_Mapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1D_Mapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2D_Mapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMean_Mapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
ReduceSum_Mapper = NodeMapper(ReduceSumParser(), BasicReduceSumBindings)
RequantShift_Mapper = NodeMapper(RequantShiftParser(), MemPoolRQSBindings_x_32_32_8)
Reshape_Mapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELU_Mapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDiv_Mapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
Softmax_Mapper = NodeMapper(iSoftmaxParser(), [BasicSoftmaxBinding])
Transpose_Mapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
Unsqueeze_Mapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

# MemPool specific bindings
Conv1D_Mapper = NodeMapper(GenericConv1DParser(), [MemPoolConv1D_8_8_32_Binding])
Conv2D_Mapper = NodeMapper(GenericConv2DParser(), [MemPoolConv2D_8_8_32_Binding])
DWConv1D_Mapper = NodeMapper(GenericDWConv1DParser(), [MemPoolDWConv1D_8_8_32_Binding])
DWConv2D_Mapper = NodeMapper(GenericDWConv2DParser(), [MemPoolDWConv2D_8_8_32_Binding])
GEMM_Mapper = NodeMapper(GenericGEMMParser(), [MemPoolGEMMBinding_8_8_32_32])
MatMul_Mapper = NodeMapper(MatMulParser(), [MemPoolMatMul_8_8_32_Binding])
MaxPool_Mapper = NodeMapper(GenericMaxPool2DParser(), [MemPoolMaxPool2D_8_8_Binding])
M1HSA_Mapper = NodeMapper(MemPoolM1HSAParser(), [MemPoolMHSA_1H_INT8_Binding])
M2HSA_Mapper = NodeMapper(MemPoolM2HSAParser(), [MemPoolMHSA_2H_INT8_Binding])
M4HSA_Mapper = NodeMapper(MemPoolITAM4HSAParser(), [MemPoolMHSA_4H_INT8_Binding])
RQMatMul_Mapper = NodeMapper(RQMatMulParser(), [MemPoolRQMatMul_8_8_32_32_Binding])
RQGemm_Mapper = NodeMapper(RQGEMMParser(), [MemPoolRQGEMMBinding_8_8_32_32_32_8])

MHSA_Mappers = [M4HSA_Mapper, M2HSA_Mapper, M1HSA_Mapper]

Conv_Mappers = [Conv2D_Mapper, DWConv2D_Mapper, Conv1D_Mapper, DWConv1D_Mapper]

SliceMapper = NodeMapper(SliceParser(), BasicSliceBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

MemPoolMapping = {
    'Add': AddLayer([Add_Mapper]),
    'Conv': ConvLayer(Conv_Mappers + GenericConv_Mappers),  # Mapper with higher priority should be placed first!
    'DebugPrint': DebugPrintLayer([DebugPrint_Mapper]),
    'Div': IntegerDivLayer([IntegerDiv_Mapper]),
    'Flatten': ReshapeLayer([Flatten_Mapper]),
    'Gather': GatherLayer([Gather_Mapper]),
    'Gemm': GEMMLayer([GEMM_Mapper]),
    'iGELU': iGELULayer([GELU_Mapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNorm_Mapper]),
    'IntegerDiv': IntegerDivLayer([IntegerDiv_Mapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMean_Mapper]),
    'iSoftmax': iSoftmaxLayer([Softmax_Mapper]),
    'ITAMax': ITAMaxLayer([ITAMaxMapper]),
    'MatMul': MatMulLayer([MatMul_Mapper]),
    'MatMulInteger': MatMulLayer([MatMul_Mapper]),
    'MaxPool': MaxPoolLayer([MaxPool_Mapper]),
    'MHSA': MHSALayer(MHSA_Mappers),
    'Mul': MulLayer([Mul_Mapper]),
    'Pad': PadLayer([Pad1D_Mapper, Pad2D_Mapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMean_Mapper]),
    'ReduceSum': ReduceSumLayer([ReduceSum_Mapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_Mapper]),
    'RequantShift': RequantShiftLayer([RequantShift_Mapper]),
    'Reshape': ReshapeLayer([Reshape_Mapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDiv_Mapper]),
    'RQGemm': RQGEMMLayer([RQGemm_Mapper]),
    'RQMatMul': RQMatMulLayer([RQMatMul_Mapper]),
    'Transpose': TransposeLayer([Transpose_Mapper]),
    'Unsqueeze': ReshapeLayer([Unsqueeze_Mapper]),
    'Slice': SliceLayer([SliceMapper])
}


class MemPoolVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.MemPoolInitTemplate
    allocTemplate = AllocateTemplate.MemPoolAllocateTemplate
    deallocTemplate = FreeTemplate.MemPoolLocalTemplate


class MemPoolTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.MemPoolInitTemplate
    allocTemplate = AllocateTemplate.MemPoolAllocateTemplate
    deallocTemplate = FreeTemplate.MemPoolLocalTemplate


class MemPoolConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.MemPoolGlobalInitTemplate
    allocTemplate = AllocateTemplate.MemPoolGlobalAllocateTemplate
    deallocTemplate = FreeTemplate.MemPoolGlobalTemplate

    def _bufferRepresentation(self) -> Dict:
        retDict = super()._bufferRepresentation()
        # WIESEP: Workaround for banshee simulations.
        # Due to problems wrongly copied bytes, we want array sized a multiple of 4
        bytes = np.prod(self.shape) * (self._type.typeWidth // 8)
        if bytes % 4 != 0:
            bytes = 4 * int((bytes / 4 + 1))
        size = (bytes * 8) // self._type.typeWidth
        retDict['size'] = int(size)
        return retDict


class MemPoolStructBuffer(StructBuffer):

    initTemplate = AllocateTemplate.MemPoolStructInitTemplate
    allocTemplate = AllocateTemplate.MemPoolStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


MemPoolOptimizer = TopologyOptimizer([
    MemPoolFuseMHSAPass(H = 8, bias = False, preSoftMaxRQ = True, integerDiv = False),
    MemPoolFuseMHSAPass(H = 1, bias = False, preSoftMaxRQ = True, integerDiv = False),
    MemPoolFuseMHSAPass(H = -1, bias = False, preSoftMaxRQ = True, integerDiv = False),
    MemPoolFuseMHSAPass(H = -1, bias = True, preSoftMaxRQ = True, integerDiv = False),
    MemPoolSplitMHSAPass(),
    iGELURequantMergePass(),
    MatMulAddMergePass(),
    SplitAddPass(),
    MergeConstAddAndRequantPass(),
    MemPoolMatMulRequantMergePass(),
    MemPoolGEMMRequantMergePass(),
    ExtractPaddingFromConvPass(),
    ExtractPaddingFromPoolPass(),
    # DebugPrintPass(r'.*[Mm]at[Mm]ul.*', position = 'after'),
])

includeList = ["DeeployMath.h", "runtime.h", "synchronization.h"]


class MemPoolEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = MemPoolMapping, initCode: str = "", includeList = includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class MemPoolPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [MemPoolEngine("MemPool")],
                 variableBuffer = MemPoolVariableBuffer,
                 constantBuffer = MemPoolConstantBuffer,
                 structBuffer = MemPoolStructBuffer,
                 transientBuffer = MemPoolTransientBuffer):
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)
