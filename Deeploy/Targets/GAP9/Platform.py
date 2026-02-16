# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NetworkContext, NodeMapper, \
    NodeTemplate, StructBuffer, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform, MemoryPlatformWrapper
from Deeploy.Targets.GAP9.Templates import AllocateTemplate, FreeTemplate
# Import GAP9-specific tiler bindings
from Deeploy.Targets.GAP9.Tiler import GAP9AddTilingReadyBindings, GAP9ConcatTilingReadyBindings, \
    GAP9Conv2DTilingReadyBindings, GAP9DWConv2DTilingReadyBindings, GAP9FlattenTilingReadyBindings, \
    GAP9FPGELUTilingReadyBindings, GAP9FPGEMMTilingReadyBindings, GAP9GatherTilingReadyBindings, \
    GAP9iHardswishTilingReadyBindings, GAP9iRMSNormTilingReadyBindings, GAP9iRQSGELUTilingReadyBindings, \
    GAP9LayernormTilingReadyBindings, GAP9MatMulTilingReadyBindings, GAP9MaxPool2DTilingReadyBindings, \
    GAP9MulTilingReadyBindings, GAP9ReduceSumTilingReadyBindings, GAP9ReluTilingReadyBindings, \
    GAP9RQAddTilingReadyBindings, GAP9RQSConv2DTilingReadyBindings, GAP9RQSDWConv2DTilingReadyBindings, \
    GAP9RQSGEMMTilingReadyBindings, GAP9RQSiHardswishTilingReadyBindings, GAP9RQSMatrixVecTilingReadyBindings, \
    GAP9RQSTallGEMMTilingReadyBindings, GAP9RQSTilingReadyBindings, GAP9SGDTilingReadyBindings, \
    GAP9SoftmaxCrossEntropyGradTilingReadyBindings, GAP9SoftmaxCrossEntropyTilingReadyBindings, \
    GAP9SoftmaxGradTilingReadyBindings, GAP9SoftmaxTilingReadyBindings, GAP9TransposeTilingReadyBindings, \
    GAP9UniformRQSTilingReadyBindings
from Deeploy.Targets.Generic.Bindings import BasicGEMMBindings, BasicPad1DBindings, BasicPad2DBindings, \
    BasicRQIntegerDivBinding
from Deeploy.Targets.Generic.Layers import AddLayer, ConcatLayer, ConvLayer, GatherLayer, GELULayer, GEMMLayer, \
    LayerNormLayer, MatMulLayer, MaxPoolLayer, MulLayer, PadLayer, QuantLayer, ReduceMeanLayer, ReduceSumLayer, \
    ReluLayer, RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, RQSiHardswishLayer, SGDLayer, \
    SliceLayer, SoftmaxCrossEntropyLossGradLayer, SoftmaxCrossEntropyLossLayer, SoftmaxGradLayer, SoftmaxLayer, \
    TransposeLayer, iHardswishLayer, iRMSNormLayer
from Deeploy.Targets.Generic.Parsers import AddParser, ConcatParser, DequantParser, FlattenParser, GatherParser, \
    GELUParser, GEMMParser, LayerNormParser, MatMulParser, MaxPool2DParser, MulParser, Pad1DParser, Pad2DParser, \
    QuantParser, ReduceMeanParser, ReduceSumParser, ReluParser, RequantShiftParser, ReshapeParser, RQAddParser, \
    RQIntegerDivParser, RQSiGELUParser, RQSiHardswishParser, SGDParser, SliceParser, \
    SoftmaxCrossEntropyLossGradParser, SoftmaxCrossEntropyLossParser, SoftmaxGradParser, SoftmaxParser, \
    TransposeParser, UniformRequantShiftParser, UnsqueezeParser, iHardswishParser, iRMSNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Targets.PULPOpen.Bindings import BasicDequantBindings, BasicQuantBindings, PULPDMASliceBindings, \
    PULPDWConv1DBinding, PULPReduceMeanBindings, PULPRQSConv1DBindings, PULPSliceBindings
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer, PULPRQSGEMMLayer
from Deeploy.Targets.PULPOpen.Parsers import PULPConv1DParser, PULPConv2DParser, PULPDWConv1DParser, \
    PULPDWConv2DParser, PULPFPConv2DParser, PULPFPDWConv2DParser, PULPGEMMParser, PULPMatrixVecParser, \
    PULPTallGEMMParser

# Create GAP9-specific NodeMappers
GAP9_RQAddMapper = NodeMapper(RQAddParser(), GAP9RQAddTilingReadyBindings)
GAP9_AddMapper = NodeMapper(AddParser(), GAP9AddTilingReadyBindings)
GAP9_FlattenMapper = NodeMapper(FlattenParser(), GAP9FlattenTilingReadyBindings)
GAP9_GELUMapper = NodeMapper(GELUParser(), GAP9FPGELUTilingReadyBindings)
GAP9_GatherMapper = NodeMapper(GatherParser(), GAP9GatherTilingReadyBindings)
GAP9_MulMapper = NodeMapper(MulParser(), GAP9MulTilingReadyBindings)
GAP9_Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
GAP9_Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
GAP9_ReshapeMapper = NodeMapper(ReshapeParser(), GAP9FlattenTilingReadyBindings)
GAP9_TransposeMapper = NodeMapper(TransposeParser(), GAP9TransposeTilingReadyBindings)
GAP9_UnsqueezeMapper = NodeMapper(UnsqueezeParser(), GAP9FlattenTilingReadyBindings)
GAP9_RequantShiftMapper = NodeMapper(RequantShiftParser(), GAP9RQSTilingReadyBindings)
GAP9_UniformRequantShiftMapper = NodeMapper(UniformRequantShiftParser(), GAP9UniformRQSTilingReadyBindings)
GAP9_ReduceMeanMapper = NodeMapper(ReduceMeanParser(), PULPReduceMeanBindings)
GAP9_ReduceSumMapper = NodeMapper(ReduceSumParser(), GAP9ReduceSumTilingReadyBindings)
GAP9_MatMulMapper = NodeMapper(MatMulParser(), GAP9MatMulTilingReadyBindings)
GAP9_RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
GAP9_RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), GAP9iRQSGELUTilingReadyBindings)
GAP9_Conv1DMapper = NodeMapper(PULPConv1DParser(), PULPRQSConv1DBindings)
GAP9_DWConv1DMapper = NodeMapper(PULPDWConv1DParser(), [PULPDWConv1DBinding])
GAP9_FPConv2DMapper = NodeMapper(PULPFPConv2DParser(), GAP9Conv2DTilingReadyBindings)
GAP9_Conv2DMapper = NodeMapper(PULPConv2DParser(), GAP9RQSConv2DTilingReadyBindings)
GAP9_FPDWConv2DMapper = NodeMapper(PULPFPDWConv2DParser(), GAP9DWConv2DTilingReadyBindings)
GAP9_DWConv2DMapper = NodeMapper(PULPDWConv2DParser(), GAP9RQSDWConv2DTilingReadyBindings)
GAP9_GEMMMapper = NodeMapper(PULPGEMMParser(), GAP9RQSGEMMTilingReadyBindings)
GAP9_FloatGEMMMapper = NodeMapper(GEMMParser(), GAP9FPGEMMTilingReadyBindings)
GAP9_MatrixVecMapper = NodeMapper(PULPMatrixVecParser(), GAP9RQSMatrixVecTilingReadyBindings)
GAP9_TallGEMMMapper = NodeMapper(PULPTallGEMMParser(), GAP9RQSTallGEMMTilingReadyBindings)
GAP9_MaxPool2DMapper = NodeMapper(MaxPool2DParser(), GAP9MaxPool2DTilingReadyBindings)
GAP9_LayerNormMapper = NodeMapper(LayerNormParser(), GAP9LayernormTilingReadyBindings)
GAP9_ReluMapper = NodeMapper(ReluParser(), GAP9ReluTilingReadyBindings)
GAP9_SoftmaxMapper = NodeMapper(SoftmaxParser(), GAP9SoftmaxTilingReadyBindings)
GAP9_SoftmaxGradMapper = NodeMapper(SoftmaxGradParser(), GAP9SoftmaxGradTilingReadyBindings)
GAP9_Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), GAP9SoftmaxTilingReadyBindings)
GAP9_ConcatMapper = NodeMapper(ConcatParser(), GAP9ConcatTilingReadyBindings)
GAP9_DMASliceMapper = NodeMapper(SliceParser(), PULPDMASliceBindings)
GAP9_SliceMapper = NodeMapper(SliceParser(), PULPSliceBindings)
GAP9_iRMSNormMapper = NodeMapper(iRMSNormParser(), GAP9iRMSNormTilingReadyBindings)
GAP9_iHardswishMapper = NodeMapper(iHardswishParser(), GAP9iHardswishTilingReadyBindings)
GAP9_RQSiHardswishMapper = NodeMapper(RQSiHardswishParser(), GAP9RQSiHardswishTilingReadyBindings)
GAP9_SoftmaxCrossEntropyLossMapper = NodeMapper(SoftmaxCrossEntropyLossParser(),
                                                GAP9SoftmaxCrossEntropyTilingReadyBindings)
GAP9_SoftmaxCrossEntropyLossGradMapper = NodeMapper(SoftmaxCrossEntropyLossGradParser(),
                                                    GAP9SoftmaxCrossEntropyGradTilingReadyBindings)
GAP9_SGDMapper = NodeMapper(SGDParser(), GAP9SGDTilingReadyBindings)
GAP9_QuantMapper = NodeMapper(QuantParser(), BasicQuantBindings)
GAP9_DequantMapper = NodeMapper(DequantParser(), BasicDequantBindings)
GAP9_GEMMDequantMapper = NodeMapper(PULPGEMMParser(), BasicGEMMBindings)

# GAP9-specific mapping using ClDma
GAP9Mapping = {
    'Conv':
        ConvLayer([GAP9_FPConv2DMapper, GAP9_FPDWConv2DMapper]),
    'RequantizedConv':
        PULPRQSConvLayer([GAP9_Conv2DMapper, GAP9_DWConv2DMapper, GAP9_Conv1DMapper, GAP9_DWConv1DMapper]),
    'RequantizedGemm':
        PULPRQSGEMMLayer([GAP9_MatrixVecMapper, GAP9_TallGEMMMapper, GAP9_GEMMMapper]),
    'Gemm':
        GEMMLayer([GAP9_FloatGEMMMapper, GAP9_GEMMDequantMapper]),
    'Gelu':
        GELULayer([GAP9_GELUMapper]),
    'LayerNormalization':
        LayerNormLayer([GAP9_LayerNormMapper]),
    'MaxPool':
        MaxPoolLayer([GAP9_MaxPool2DMapper]),
    'RequantizediGELU':
        RQSiGELULayer([GAP9_RQGELU_int8_Mapper]),
    'RQIntegerDiv':
        RQIntegerDivLayer([GAP9_RQIntegerDivMapper]),
    'MatMul':
        MatMulLayer([GAP9_MatMulMapper]),
    'IntegerMean':
        ReduceMeanLayer([GAP9_ReduceMeanMapper]),
    'iSoftmax':
        SoftmaxLayer([GAP9_Softmax_int8_Mapper]),
    'Softmax':
        SoftmaxLayer([GAP9_SoftmaxMapper]),
    'ReduceMean':
        ReduceMeanLayer([GAP9_ReduceMeanMapper]),
    'ReduceSum':
        ReduceSumLayer([GAP9_ReduceSumMapper]),
    'RequantShift':
        RequantShiftLayer([GAP9_UniformRequantShiftMapper, GAP9_RequantShiftMapper]),
    'Add':
        AddLayer([GAP9_AddMapper]),
    'Flatten':
        ReshapeLayer([GAP9_FlattenMapper]),
    'Gather':
        GatherLayer([GAP9_GatherMapper]),
    'Mul':
        MulLayer([GAP9_MulMapper]),
    'Pad':
        PadLayer([GAP9_Pad1DMapper, GAP9_Pad2DMapper]),
    'Relu':
        ReluLayer([GAP9_ReluMapper]),
    'Reshape':
        ReshapeLayer([GAP9_ReshapeMapper]),
    'Squeeze':
        ReshapeLayer([GAP9_UnsqueezeMapper]),
    'Transpose':
        TransposeLayer([GAP9_TransposeMapper]),
    'Unsqueeze':
        ReshapeLayer([GAP9_UnsqueezeMapper]),
    'Slice':
        SliceLayer([GAP9_SliceMapper, GAP9_DMASliceMapper]),
    'RequantizedAdd':
        AddLayer([GAP9_RQAddMapper]),
    'Concat':
        ConcatLayer([GAP9_ConcatMapper]),
    'iRMSNorm':
        iRMSNormLayer([GAP9_iRMSNormMapper]),
    'iHardswish':
        iHardswishLayer([GAP9_iHardswishMapper]),
    'RequantizediHardswish':
        RQSiHardswishLayer([GAP9_RQSiHardswishMapper]),
    'Quant':
        QuantLayer([GAP9_QuantMapper]),
    'Dequant':
        QuantLayer([GAP9_DequantMapper]),
    'SoftmaxGrad':
        SoftmaxGradLayer([GAP9_SoftmaxGradMapper]),
    'SoftmaxCrossEntropyLoss':
        SoftmaxCrossEntropyLossLayer([GAP9_SoftmaxCrossEntropyLossMapper]),
    'SoftmaxCrossEntropyLossGrad':
        SoftmaxCrossEntropyLossGradLayer([GAP9_SoftmaxCrossEntropyLossGradMapper]),
    'SGD':
        SGDLayer([GAP9_SGDMapper])
}


class GAP9VariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.gap9L2InitTemplate
    # allocTemplate = AllocateTemplate.gap9L2AllocateTemplate
    # deallocTemplate = FreeTemplate.gap9L2LocalTemplate

    allocTemplate = AllocateTemplate.gap9GenericAllocate
    deallocTemplate = FreeTemplate.gap9GenericFree

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {
            "type": self._instance,
            "name": self.name,
            "size": int(np.prod(self.shape)),
            "_memoryLevel": memoryLevel
        }


class GAP9TransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.gap9L2InitTemplate
    allocTemplate = AllocateTemplate.gap9GenericAllocate
    deallocTemplate = FreeTemplate.gap9GenericFree

    # allocTemplate = AllocateTemplate.gap9L2AllocateTemplate
    # deallocTemplate = FreeTemplate.gap9L2GlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}


class GAP9ConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.gap9GenericGlobalInitTemplate
    allocTemplate = AllocateTemplate.gap9L2GlobalAllocateTemplate
    deallocTemplate = FreeTemplate.gap9L2GlobalTemplate

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class GAP9StructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


_includeList = ["pmsis.h", "DeeployGAP9Math.h", "pulp_nn_kernels.h", "DeeployMchan.h"]


class GAP9ClusterEngine(DeploymentEngine):

    def __init__(self,
                 name: str,
                 Mapping = GAP9Mapping,
                 initCode = "",
                 includeList = _includeList,
                 n_cores: int = 8) -> None:
        super().__init__(name, Mapping, initCode, includeList)
        self.n_cores = n_cores


class GAP9Platform(DeploymentPlatform):

    def __init__(self,
                 engines = [GAP9ClusterEngine("GAP9Cluster")],
                 variableBuffer = GAP9VariableBuffer,
                 constantBuffer = GAP9ConstantBuffer,
                 structBuffer = GAP9StructBuffer,
                 transientBuffer = GAP9TransientBuffer) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryGAP9Platform(MemoryPlatform):

    untiledOps = ["add"]

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 engines = [GAP9ClusterEngine("GAP9Cluster")],
                 variableBuffer = GAP9VariableBuffer,
                 constantBuffer = GAP9ConstantBuffer,
                 structBuffer = GAP9StructBuffer,
                 transientBuffer = GAP9TransientBuffer) -> None:
        super().__init__(memoryHierarchy, defaultTargetMemoryLevel, engines, variableBuffer, constantBuffer,
                         structBuffer, transientBuffer)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if node.op in self.untiledOps:
            return ctxt.lookup(tensorName)._memoryLevel
        return super().getTargetMemoryLevel(node, tensorName, ctxt)


class MemoryGAP9PlatformWrapper(MemoryPlatformWrapper):

    untiledOps = []

    def __init__(self, platform: GAP9Platform, memoryHierarchy: MemoryHierarchy, defaultTargetMemoryLevel: MemoryLevel):
        assert isinstance(platform, GAP9Platform), \
        f"Given platform is not an instance of GAP9Platform. Platform type: {type(platform).__name__}"
        super().__init__(platform, memoryHierarchy, defaultTargetMemoryLevel)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if node.op in self.untiledOps:
            return ctxt.lookup(tensorName)._memoryLevel
        return super().getTargetMemoryLevel(node, tensorName, ctxt)
