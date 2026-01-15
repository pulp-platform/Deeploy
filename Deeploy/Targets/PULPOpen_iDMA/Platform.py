# SPDX-FileCopyrightText: 2023 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import NetworkContext, NodeMapper
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.Targets.Generic.Bindings import BasicGEMMBindings, BasicPad1DBindings, BasicPad2DBindings, \
    BasicRQIntegerDivBinding
from Deeploy.Targets.Generic.Layers import AddLayer, ConcatLayer, ConvLayer, GatherLayer, GELUGradLayer, GELULayer, \
    GEMMLayer, LayerNormGradLayer, LayerNormLayer, MatMulLayer, MaxPoolLayer, MulLayer, PadLayer, QuantLayer, \
    ReduceMeanLayer, ReduceSumLayer, ReluLayer, RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, \
    RQSiHardswishLayer, SGDLayer, SliceLayer, SoftmaxCrossEntropyLossGradLayer, SoftmaxCrossEntropyLossLayer, \
    SoftmaxGradLayer, SoftmaxLayer, TransposeLayer, iHardswishLayer, iRMSNormLayer
from Deeploy.Targets.Generic.Parsers import AddParser, ConcatParser, DequantParser, FlattenParser, GatherParser, \
    GELUGradParser, GELUParser, GEMMParser, LayerNormGradParser, LayerNormParser, MatMulParser, MaxPool1DParser, \
    MaxPool2DParser, MulParser, Pad1DParser, Pad2DParser, QuantParser, ReduceSumParser, ReluParser, \
    RequantShiftParser, ReshapeParser, RQAddParser, RQIntegerDivParser, RQSiGELUParser, RQSiHardswishParser, \
    SGDParser, SliceParser, SoftmaxCrossEntropyLossGradParser, SoftmaxCrossEntropyLossParser, SoftmaxGradParser, \
    SoftmaxParser, TransposeParser, UniformRequantShiftParser, UnsqueezeParser, iHardswishParser, iRMSNormParser, \
    iSoftmaxParser
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer, PULPRQSGEMMLayer
from Deeploy.Targets.PULPOpen.Parsers import PULPConv1DParser, PULPConv2DParser, PULPDWConv1DParser, \
    PULPDWConv2DParser, PULPFPConv2DParser, PULPFPDWConv2DParser, PULPGEMMParser, PULPMatrixVecParser, \
    PULPReduceMeanParser, PULPTallGEMMParser
from Deeploy.Targets.PULPOpen.Platform import MemoryPULPPlatform, PULPClusterEngine, PULPConstantBuffer, PULPPlatform, \
    PULPStructBuffer, PULPTransientBuffer, PULPVariableBuffer
from Deeploy.Targets.PULPOpen_iDMA.Bindings import BasicDequantBindings, BasicQuantBindings, PULPDMASliceBindings, \
    PULPDWConv1DBinding
from Deeploy.Targets.PULPOpen_iDMA.Tiler import PULPAddTilingReadyBindings, PULPConcatTilingReadyBindings, \
    PULPConv2DTilingReadyBindings, PULPDWConv2DTilingReadyBindings, PULPFlattenTilingReadyBindings, \
    PULPFPGELUGradTilingReadyBindings, PULPFPGELUTilingReadyBindings, PULPFPGEMMTilingReadyBindings, \
    PULPGatherTilingReadyBindings, PULPiHardswishTilingReadyBindings, PULPiRMSNormTilingReadyBindings, \
    PULPiRQSGELUTilingReadyBindings, PULPLayernormGradTilingReadyBindings, PULPLayernormTilingReadyBindings, \
    PULPMatMulTilingReadyBindings, PULPMaxPool1DTilingReadyBindings, PULPMaxPool2DTilingReadyBindings, \
    PULPMulTilingReadyBindings, PULPReduceMeanTilingReadyBindings, PULPReduceSumTilingReadyBindings, \
    PULPReluTilingReadyBindings, PULPRQAddTilingReadyBindings, PULPRQSConv1DTilingReadyBindings, \
    PULPRQSConv2DTilingReadyBindings, PULPRQSDWConv2DTilingReadyBindings, PULPRQSGEMMTilingReadyBindings, \
    PULPRQSiHardswishTilingReadyBindings, PULPRQSMatrixVecTilingReadyBindings, PULPRQSTallGEMMTilingReadyBindings, \
    PULPRQSTilingReadyBindings, PULPSGDTilingReadyBindings, PULPSliceTilingReadyBindings, \
    PULPSoftmaxCrossEntropyGradTilingReadyBindings, PULPSoftmaxCrossEntropyTilingReadyBindings, \
    PULPSoftmaxGradTilingReadyBindings, PULPSoftmaxTilingReadyBindings, PULPTransposeTilingReadyBindings, \
    PULPUniformRQSTilingReadyBindings

RQAddMapper = NodeMapper(RQAddParser(), PULPRQAddTilingReadyBindings)
AddMapper = NodeMapper(AddParser(), PULPAddTilingReadyBindings)
FlattenMapper = NodeMapper(FlattenParser(), PULPFlattenTilingReadyBindings)
GELUMapper = NodeMapper(GELUParser(), PULPFPGELUTilingReadyBindings)
GELUGradMapper = NodeMapper(GELUGradParser(), PULPFPGELUGradTilingReadyBindings)
GatherMapper = NodeMapper(GatherParser(), PULPGatherTilingReadyBindings)
MulMapper = NodeMapper(MulParser(), PULPMulTilingReadyBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), PULPFlattenTilingReadyBindings)
TransposeMapper = NodeMapper(TransposeParser(), PULPTransposeTilingReadyBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), PULPFlattenTilingReadyBindings)

RequantShiftMapper = NodeMapper(RequantShiftParser(), PULPRQSTilingReadyBindings)
UniformRequantShiftMapper = NodeMapper(UniformRequantShiftParser(), PULPUniformRQSTilingReadyBindings)

ReduceMeanMapper = NodeMapper(PULPReduceMeanParser(), PULPReduceMeanTilingReadyBindings)
ReduceSumMapper = NodeMapper(ReduceSumParser(), PULPReduceSumTilingReadyBindings)
MatMulMapper = NodeMapper(MatMulParser(), PULPMatMulTilingReadyBindings)
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), PULPiRQSGELUTilingReadyBindings)

Conv1DMapper = NodeMapper(PULPConv1DParser(), PULPRQSConv1DTilingReadyBindings)
DWConv1DMapper = NodeMapper(PULPDWConv1DParser(), [PULPDWConv1DBinding])
FPConv2DMapper = NodeMapper(PULPFPConv2DParser(), PULPConv2DTilingReadyBindings)
Conv2DMapper = NodeMapper(PULPConv2DParser(), PULPRQSConv2DTilingReadyBindings)
FPDWConv2DMapper = NodeMapper(PULPFPDWConv2DParser(), PULPDWConv2DTilingReadyBindings)
DWConv2DMapper = NodeMapper(PULPDWConv2DParser(), PULPRQSDWConv2DTilingReadyBindings)
GEMMMapper = NodeMapper(PULPGEMMParser(), PULPRQSGEMMTilingReadyBindings)
FloatGEMMMapper = NodeMapper(GEMMParser(), PULPFPGEMMTilingReadyBindings)
MatrixVecMapper = NodeMapper(PULPMatrixVecParser(), PULPRQSMatrixVecTilingReadyBindings)
TallGEMMMapper = NodeMapper(PULPTallGEMMParser(), PULPRQSTallGEMMTilingReadyBindings)
MaxPool1DMapper = NodeMapper(MaxPool1DParser(), PULPMaxPool1DTilingReadyBindings)
MaxPool2DMapper = NodeMapper(MaxPool2DParser(), PULPMaxPool2DTilingReadyBindings)
LayerNormMapper = NodeMapper(LayerNormParser(), PULPLayernormTilingReadyBindings)
LayerNormGradMapper = NodeMapper(LayerNormGradParser(), PULPLayernormGradTilingReadyBindings)
ReluMapper = NodeMapper(ReluParser(), PULPReluTilingReadyBindings)
SoftmaxMapper = NodeMapper(SoftmaxParser(), PULPSoftmaxTilingReadyBindings)
SoftmaxGradMapper = NodeMapper(SoftmaxGradParser(), PULPSoftmaxGradTilingReadyBindings)
Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), PULPSoftmaxTilingReadyBindings)

ConcatMapper = NodeMapper(ConcatParser(), PULPConcatTilingReadyBindings)

DMASliceMapper = NodeMapper(SliceParser(), PULPDMASliceBindings)

SliceMapper = NodeMapper(SliceParser(), PULPSliceTilingReadyBindings)

iRMSNormMapper = NodeMapper(iRMSNormParser(), PULPiRMSNormTilingReadyBindings)

iHardswishMapper = NodeMapper(iHardswishParser(), PULPiHardswishTilingReadyBindings)
RQSiHardswishMapper = NodeMapper(RQSiHardswishParser(), PULPRQSiHardswishTilingReadyBindings)
SoftmaxCrossEntropyLossMapper = NodeMapper(SoftmaxCrossEntropyLossParser(), PULPSoftmaxCrossEntropyTilingReadyBindings)
SoftmaxCrossEntropyLossGradMapper = NodeMapper(SoftmaxCrossEntropyLossGradParser(),
                                               PULPSoftmaxCrossEntropyGradTilingReadyBindings)
SGDMapper = NodeMapper(SGDParser(), PULPSGDTilingReadyBindings)
QuantMapper = NodeMapper(QuantParser(), BasicQuantBindings)
DequantMapper = NodeMapper(DequantParser(), BasicDequantBindings)
GEMMDequantMapper = NodeMapper(PULPGEMMParser(), BasicGEMMBindings)
PULPMapping = {
    'Conv': ConvLayer([FPConv2DMapper, FPDWConv2DMapper]),
    'RequantizedConv': PULPRQSConvLayer([Conv2DMapper, DWConv2DMapper, Conv1DMapper, DWConv1DMapper]),
    'RequantizedGemm': PULPRQSGEMMLayer([MatrixVecMapper, TallGEMMMapper, GEMMMapper]),
    'Gemm': GEMMLayer([FloatGEMMMapper, GEMMDequantMapper]),
    'Gelu': GELULayer([GELUMapper]),
    'GeluGrad': GELUGradLayer([GELUGradMapper]),
    'LayerNormalization': LayerNormLayer([LayerNormMapper]),
    'LayerNormalizationGrad': LayerNormGradLayer([LayerNormGradMapper]),
    'MaxPool': MaxPoolLayer([MaxPool1DMapper, MaxPool2DMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_int8_Mapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': SoftmaxLayer([Softmax_int8_Mapper]),
    'Softmax': SoftmaxLayer([SoftmaxMapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'ReduceSum': ReduceSumLayer([ReduceSumMapper]),
    'RequantShift': RequantShiftLayer([UniformRequantShiftMapper, RequantShiftMapper]),
    'Add': AddLayer([AddMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Relu': ReluLayer([ReluMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'Squeeze': ReshapeLayer([UnsqueezeMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Slice': SliceLayer([SliceMapper, DMASliceMapper]),
    'RequantizedAdd': AddLayer([RQAddMapper]),
    'Concat': ConcatLayer([ConcatMapper]),
    'iRMSNorm': iRMSNormLayer([iRMSNormMapper]),
    'iHardswish': iHardswishLayer([iHardswishMapper]),
    'RequantizediHardswish': RQSiHardswishLayer([RQSiHardswishMapper]),
    'Quant': QuantLayer([QuantMapper]),
    'Dequant': QuantLayer([DequantMapper]),
    'SoftmaxGrad': SoftmaxGradLayer([SoftmaxGradMapper]),
    'SoftmaxCrossEntropyLoss': SoftmaxCrossEntropyLossLayer([SoftmaxCrossEntropyLossMapper]),
    'SoftmaxCrossEntropyLossGrad': SoftmaxCrossEntropyLossGradLayer([SoftmaxCrossEntropyLossGradMapper]),
    'SGD': SGDLayer([SGDMapper])
}

# SCHEREMO: stdint is included before pulp_nn_kernels.h because it is supposed to be included in there, but isn't...
_includeList = [
    "pmsis.h", "stdint.h", "pulp_nn_kernels.h", "DeeployBasicMath.h", "DeeployPULPMath.h", "bsp/ram.h", "pulp_core.h"
]


class PULPClusterEngine_iDMA(PULPClusterEngine):

    def __init__(self,
                 name: str,
                 Mapping = PULPMapping,
                 initCode = "",
                 includeList = _includeList,
                 n_cores: int = 8) -> None:
        super().__init__(name, Mapping, initCode, includeList, n_cores)


class PULPPlatform_iDMA(PULPPlatform):

    def __init__(self,
                 engines = [PULPClusterEngine_iDMA("PULPCluster")],
                 variableBuffer = PULPVariableBuffer,
                 constantBuffer = PULPConstantBuffer,
                 structBuffer = PULPStructBuffer,
                 transientBuffer = PULPTransientBuffer) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryPULPPlatform_iDMA(MemoryPULPPlatform):

    untiledOps = ["add"]

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 engines = [PULPClusterEngine_iDMA("PULPCluster")],
                 variableBuffer = PULPVariableBuffer,
                 constantBuffer = PULPConstantBuffer,
                 structBuffer = PULPStructBuffer,
                 transientBuffer = PULPTransientBuffer) -> None:
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if node.op in self.untiledOps:
            return ctxt.lookup(tensorName)._memoryLevel
        return super().getTargetMemoryLevel(node, tensorName, ctxt)