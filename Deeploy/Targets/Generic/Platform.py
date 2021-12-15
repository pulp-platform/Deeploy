# ----------------------------------------------------------------------
#
# File: GenericPlatform.py
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

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NodeMapper, NodeTemplate, \
    StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.Targets.Generic.Bindings import BasicAddBindings, BasicConv1DBinding, BasicConv2DBinding, \
    BasicDebugPrintBindings, BasicDWConv1DBinding, BasicDWConv2DBinding, BasicGatherBindings, BasicGELUBinding, \
    BasicGEMMBinding, BasicIntegerDivBinding, BasicITAPartialSoftmaxBinding, BasicITASoftmaxBinding, \
    BasicLayerNormBinding, BasicMatMulBinding, BasicMaxPool2DBinding, BasicMulBindings, BasicPad1DBindings, \
    BasicPad2DBindings, BasicReduceMeanBindings, BasicReduceSumBindings, BasicReshapeBindings, \
    BasicRQIntegerDivBinding, BasicRQSBindings, BasicRQSGELUBinding, BasicSliceBindings, BasicSoftmaxBinding, \
    BasicTransposeBindings, DummyBinding
from Deeploy.Targets.Generic.Layers import AddLayer, ConvLayer, DebugPrintLayer, GatherLayer, GEMMLayer, \
    IntegerDivLayer, ITAMaxLayer, MatMulLayer, MaxPoolLayer, MulLayer, PadLayer, ReduceMeanLayer, ReduceSumLayer, \
    RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, SliceLayer, TransposeLayer, iGELULayer, \
    iLayerNormLayer, iSoftmaxLayer
from Deeploy.Targets.Generic.Parsers import AddParser, DebugParser, DummyParser, FlattenParser, GatherParser, \
    GenericConv1DParser, GenericConv2DParser, GenericDWConv1DParser, GenericDWConv2DParser, GenericGEMMParser, \
    GenericMaxPool2DParser, IntegerDivParser, ITAMaxParser, ITAPartialMaxParser, MatMulParser, MulParser, Pad1DParser, \
    Pad2DParser, ReduceMeanParser, ReduceSumParser, RequantShiftParser, ReshapeParser, RQIntegerDivParser, \
    RQSiGELUParser, SliceParser, TransposeParser, UnsqueezeParser, iGELUParser, iLayerNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import ExtractPaddingFromConvPass, \
    ExtractPaddingFromPoolPass, MatMulAddMergePass, MergeConstAddAndRequantPass, iGELURequantMergePass

AddMapper = NodeMapper(AddParser(), BasicAddBindings)
Conv1DMapper = NodeMapper(GenericConv1DParser(), [BasicConv1DBinding])
Conv2DMapper = NodeMapper(GenericConv2DParser(), [BasicConv2DBinding])
DebugMapper = NodeMapper(DebugParser(), BasicDebugPrintBindings)
DWConv1DMapper = NodeMapper(GenericDWConv1DParser(), [BasicDWConv1DBinding])
DWConv2DMapper = NodeMapper(GenericDWConv2DParser(), [BasicDWConv2DBinding])
FlattenMapper = NodeMapper(FlattenParser(), BasicReshapeBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
GELUMapper = NodeMapper(iGELUParser(), [BasicGELUBinding])
GEMMMapper = NodeMapper(GenericGEMMParser(), [BasicGEMMBinding])
iLayerNormMapper = NodeMapper(iLayerNormParser(), [BasicLayerNormBinding])
IntegerDivMapper = NodeMapper(IntegerDivParser(), [BasicIntegerDivBinding])
ITAMaxMapper = NodeMapper(ITAMaxParser(), [BasicITASoftmaxBinding])
ITAPartialMaxMapper = NodeMapper(ITAPartialMaxParser(), [BasicITAPartialSoftmaxBinding])
MatMulMapper = NodeMapper(MatMulParser(), [BasicMatMulBinding])
MaxPoolMapper = NodeMapper(GenericMaxPool2DParser(), [BasicMaxPool2DBinding])
MulMapper = NodeMapper(MulParser(), BasicMulBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReduceMeanMapper = NodeMapper(ReduceMeanParser(), BasicReduceMeanBindings)
ReduceSumMapper = NodeMapper(ReduceSumParser(), BasicReduceSumBindings)
RequantShiftMapper = NodeMapper(RequantShiftParser(), BasicRQSBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), BasicReshapeBindings)
RQGELUMapper = NodeMapper(RQSiGELUParser(), [BasicRQSGELUBinding])
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
SoftmaxMapper = NodeMapper(iSoftmaxParser(), [BasicSoftmaxBinding])
TransposeMapper = NodeMapper(TransposeParser(), BasicTransposeBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

SliceMapper = NodeMapper(SliceParser(), BasicSliceBindings)

# Dummy nodes are intended for development purposes only!
# They should always generate compiler errors to not accidentally end up in production code
DummyMapper = NodeMapper(DummyParser(), [DummyBinding])

GenericMapping = {
    'Add': AddLayer([AddMapper]),
    'Conv': ConvLayer([Conv2DMapper, DWConv2DMapper, Conv1DMapper, DWConv1DMapper]),
    'DebugPrint': DebugPrintLayer([DebugMapper]),
    'Div': IntegerDivLayer([IntegerDivMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Gemm': GEMMLayer([GEMMMapper]),
    'iGELU': iGELULayer([GELUMapper]),
    'iLayerNorm': iLayerNormLayer([iLayerNormMapper]),
    'IntegerDiv': IntegerDivLayer([IntegerDivMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': iSoftmaxLayer([SoftmaxMapper]),
    'ITAMax': ITAMaxLayer([ITAMaxMapper]),
    'ITAPartialMax': ITAMaxLayer([ITAPartialMaxMapper]),
    'MatMul': GEMMLayer([MatMulMapper]),
    'MatMulInteger': MatMulLayer([MatMulMapper]),
    'MaxPool': MaxPoolLayer([MaxPoolMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'ReduceSum': ReduceSumLayer([ReduceSumMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELUMapper]),
    'RequantShift': RequantShiftLayer([RequantShiftMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Slice': SliceLayer([SliceMapper])
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


GenericOptimizer = TopologyOptimizer([
    iGELURequantMergePass(),
    MatMulAddMergePass(),
    MergeConstAddAndRequantPass(),
    ExtractPaddingFromConvPass(),
    ExtractPaddingFromPoolPass(),
    # DebugPrintPass(r'.*[Mm]at[Mm]ul.*', position = 'after'),
])

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
