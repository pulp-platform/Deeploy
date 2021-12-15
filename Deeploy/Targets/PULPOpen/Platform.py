# ----------------------------------------------------------------------
#
# File: PULPPlatform.py
#
# Last edited: 07.03.2023
#
# Copyright (C) 2023, ETH Zurich and University of Bologna.
#
# Authors:
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

import numpy as np
import onnx_graphsurgeon as gs

from Deeploy.DeeployTypes import ConstantBuffer, DeploymentEngine, DeploymentPlatform, NetworkContext, NodeMapper, \
    NodeTemplate, StructBuffer, TopologyOptimizer, TransientBuffer, VariableBuffer
from Deeploy.MemoryLevelExtension.MemoryLevels import MemoryHierarchy, MemoryLevel
from Deeploy.MemoryLevelExtension.NetworkDeployers.MemoryLevelDeployer import MemoryPlatform, MemoryPlatformWrapper
from Deeploy.Targets.CortexM.Parsers import CMSISMaxPool2DParser
from Deeploy.Targets.Generic.Bindings import BasicGatherBindings, BasicPad1DBindings, BasicPad2DBindings, \
    BasicReshapeBindings, BasicRQIntegerDivBinding
from Deeploy.Targets.Generic.Layers import AddLayer, ConcatLayer, GatherLayer, MatMulLayer, MaxPoolLayer, MulLayer, \
    PadLayer, ReduceMeanLayer, RequantShiftLayer, ReshapeLayer, RQIntegerDivLayer, RQSiGELULayer, RQSiHardswishLayer, \
    SliceLayer, TransposeLayer, iHardswishLayer, iRMSNormLayer, iSoftmaxLayer
from Deeploy.Targets.Generic.Parsers import AddParser, ConcatParser, FlattenParser, GatherParser, MatMulParser, \
    MulParser, Pad1DParser, Pad2DParser, ReduceMeanParser, RequantShiftParser, ReshapeParser, RQIntegerDivParser, \
    RQSiGELUParser, RQSiHardswishParser, SliceParser, TransposeParser, UniformRequantShiftParser, UnsqueezeParser, \
    iHardswishParser, iRMSNormParser, iSoftmaxParser
from Deeploy.Targets.Generic.Templates import AllocateTemplate as BasicAllocateTemplate
from Deeploy.Targets.Generic.TopologyOptimizationPasses.Passes import IntegerDivRequantMergePass, \
    MergeConstAddAndRequantPass, MergeTrueIntegerDivRequantShiftPass, RQSSplitPass, SkipEmptyConcatPass, \
    SkipUnityRequantPass, iGELURequantMergePass, iHardswishRequantMergePass
from Deeploy.Targets.PULPOpen.Bindings import PULPConv1DBinding, PULPDMASliceBindings, PULPDWConv1DBinding, \
    PULPReduceMeanBindings
from Deeploy.Targets.PULPOpen.Layers import PULPRQSConvLayer, PULPRQSGEMMLayer
from Deeploy.Targets.PULPOpen.Parsers import PULPConv1DParser, PULPConv2DParser, PULPDWConv1DParser, \
    PULPDWConv2DParser, PULPGEMMParser, PULPMatrixVecParser, PULPRQAddParser, PULPTallGEMMParser
from Deeploy.Targets.PULPOpen.Templates import AllocateTemplate, FreeTemplate
from Deeploy.Targets.PULPOpen.Tiler import PULPAddTilingReadyBindings, PULPConcatTilingReadyBindings, \
    PULPFlattenTilingReadyBindings, PULPiHardswishTilingReadyBindings, PULPiRMSNormTilingReadyBindings, \
    PULPiRQSGELUTilingReadyBindings, PULPiSoftmaxTilingReadyBindings, PULPMatMulTilingReadyBindings, \
    PULPMaxPool2DTilingReadyBindings, PULPMulTilingReadyBindings, PULPRQAddTilingReadyBindings, \
    PULPRQSConv2DTilingReadyBindings, PULPRQSDWConv2DTilingReadyBindings, PULPRQSGEMMTilingReadyBindings, \
    PULPRQSiHardswishTilingReadyBindings, PULPRQSMatrixVecTilingReadyBindings, PULPRQSTallGEMMTilingReadyBindings, \
    PULPRQSTilingReadyBindings, PULPTransposeTilingReadyBindings, PULPUniformRQSTilingReadyBindings
from Deeploy.Targets.PULPOpen.TopologyOptimizationPasses.Passes import PULPAddRequantMergePass, \
    PULPConvRequantMergePass, PULPGEMMRequantMergePass, PULPMatMulRequantMergePass

RQAddMapper = NodeMapper(PULPRQAddParser(), PULPRQAddTilingReadyBindings)
AddMapper = NodeMapper(AddParser(), PULPAddTilingReadyBindings)
FlattenMapper = NodeMapper(FlattenParser(), PULPFlattenTilingReadyBindings)
GatherMapper = NodeMapper(GatherParser(), BasicGatherBindings)
MulMapper = NodeMapper(MulParser(), PULPMulTilingReadyBindings)
Pad1DMapper = NodeMapper(Pad1DParser(), BasicPad1DBindings)
Pad2DMapper = NodeMapper(Pad2DParser(), BasicPad2DBindings)
ReshapeMapper = NodeMapper(ReshapeParser(), PULPFlattenTilingReadyBindings)
TransposeMapper = NodeMapper(TransposeParser(), PULPTransposeTilingReadyBindings)
UnsqueezeMapper = NodeMapper(UnsqueezeParser(), BasicReshapeBindings)

RequantShiftMapper = NodeMapper(RequantShiftParser(), PULPRQSTilingReadyBindings)
UniformRequantShiftMapper = NodeMapper(UniformRequantShiftParser(), PULPUniformRQSTilingReadyBindings)

ReduceMeanMapper = NodeMapper(ReduceMeanParser(), PULPReduceMeanBindings)
MatMulMapper = NodeMapper(MatMulParser(), PULPMatMulTilingReadyBindings)
RQIntegerDivMapper = NodeMapper(RQIntegerDivParser(), [BasicRQIntegerDivBinding])
RQGELU_int8_Mapper = NodeMapper(RQSiGELUParser(), PULPiRQSGELUTilingReadyBindings)

Conv1DMapper = NodeMapper(PULPConv1DParser(), [PULPConv1DBinding])
DWConv1DMapper = NodeMapper(PULPDWConv1DParser(), [PULPDWConv1DBinding])

Conv2DMapper = NodeMapper(PULPConv2DParser(), PULPRQSConv2DTilingReadyBindings)
DWConv2DMapper = NodeMapper(PULPDWConv2DParser(), PULPRQSDWConv2DTilingReadyBindings)
GEMMMapper = NodeMapper(PULPGEMMParser(), PULPRQSGEMMTilingReadyBindings)
MatrixVecMapper = NodeMapper(PULPMatrixVecParser(), PULPRQSMatrixVecTilingReadyBindings)
TallGEMMMapper = NodeMapper(PULPTallGEMMParser(), PULPRQSTallGEMMTilingReadyBindings)
MaxPool2DMapper = NodeMapper(CMSISMaxPool2DParser(), PULPMaxPool2DTilingReadyBindings)
Softmax_int8_Mapper = NodeMapper(iSoftmaxParser(), PULPiSoftmaxTilingReadyBindings)

ConcatMapper = NodeMapper(ConcatParser(), PULPConcatTilingReadyBindings)

SliceMapper = NodeMapper(SliceParser(), PULPDMASliceBindings)

iRMSNormMapper = NodeMapper(iRMSNormParser(), PULPiRMSNormTilingReadyBindings)

iHardswishMapper = NodeMapper(iHardswishParser(), PULPiHardswishTilingReadyBindings)
RQSiHardswishMapper = NodeMapper(RQSiHardswishParser(), PULPRQSiHardswishTilingReadyBindings)

PULPMapping = {
    'RequantizedConv': PULPRQSConvLayer([Conv2DMapper, DWConv2DMapper, Conv1DMapper, DWConv1DMapper]),
    'RequantizedGemm': PULPRQSGEMMLayer([MatrixVecMapper, TallGEMMMapper, GEMMMapper]),
    'MaxPool': MaxPoolLayer([MaxPool2DMapper]),
    'RequantizediGELU': RQSiGELULayer([RQGELU_int8_Mapper]),
    'RQIntegerDiv': RQIntegerDivLayer([RQIntegerDivMapper]),
    'MatMul': MatMulLayer([MatMulMapper]),
    'IntegerMean': ReduceMeanLayer([ReduceMeanMapper]),
    'iSoftmax': iSoftmaxLayer([Softmax_int8_Mapper]),
    'ReduceMean': ReduceMeanLayer([ReduceMeanMapper]),
    'RequantShift': RequantShiftLayer([UniformRequantShiftMapper, RequantShiftMapper]),
    'Add': AddLayer([AddMapper]),
    'Flatten': ReshapeLayer([FlattenMapper]),
    'Gather': GatherLayer([GatherMapper]),
    'Mul': MulLayer([MulMapper]),
    'Pad': PadLayer([Pad1DMapper, Pad2DMapper]),
    'Reshape': ReshapeLayer([ReshapeMapper]),
    'Transpose': TransposeLayer([TransposeMapper]),
    'Unsqueeze': ReshapeLayer([UnsqueezeMapper]),
    'Slice': SliceLayer([SliceMapper]),
    'RequantizedAdd': AddLayer([RQAddMapper]),
    'Concat': ConcatLayer([ConcatMapper]),
    'iRMSNorm': iRMSNormLayer([iRMSNormMapper]),
    'iHardswish': iHardswishLayer([iHardswishMapper]),
    'RequantizediHardswish': RQSiHardswishLayer([RQSiHardswishMapper])
}


class PULPVariableBuffer(VariableBuffer):

    initTemplate = AllocateTemplate.pulpL2InitTemplate
    # allocTemplate = AllocateTemplate.pulpL2AllocateTemplate
    # deallocTemplate = FreeTemplate.pulpL2LocalTemplate

    allocTemplate = AllocateTemplate.pulpGenericAllocate
    deallocTemplate = FreeTemplate.pulpGenericFree

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


class PULPTransientBuffer(TransientBuffer):

    initTemplate = AllocateTemplate.pulpL2InitTemplate
    allocTemplate = AllocateTemplate.pulpGenericAllocate
    deallocTemplate = FreeTemplate.pulpGenericFree

    # allocTemplate = AllocateTemplate.pulpL2AllocateTemplate
    # deallocTemplate = FreeTemplate.pulpL2GlobalTemplate

    def _bufferRepresentation(self):

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        return {"type": self._type, "name": self.name, "size": self.size, "_memoryLevel": memoryLevel}


class PULPConstantBuffer(ConstantBuffer):

    initTemplate = AllocateTemplate.pulpGenericGlobalInitTemplate
    allocTemplate = AllocateTemplate.pulpL2GlobalAllocateTemplate
    deallocTemplate = FreeTemplate.pulpL2GlobalTemplate

    def _bufferRepresentation(self):
        operatorRepresentation = super()._bufferRepresentation()

        if hasattr(self, "_memoryLevel"):
            memoryLevel = self._memoryLevel
        else:
            memoryLevel = None

        operatorRepresentation["_memoryLevel"] = memoryLevel

        return operatorRepresentation


class PULPStructBuffer(StructBuffer):

    initTemplate = BasicAllocateTemplate.referenceStructInitTemplate
    allocTemplate = BasicAllocateTemplate.referenceStructAllocateTemplate
    deallocTemplate = NodeTemplate("")


PULPOptimizer = TopologyOptimizer([
    SkipEmptyConcatPass(),
    SkipUnityRequantPass(previous_op_regex = "Concat", num_inputs = 2),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    SkipUnityRequantPass(previous_op_regex = "Reshape|Transpose", num_inputs = 1),
    RQSSplitPass(),
    MergeTrueIntegerDivRequantShiftPass(),
    IntegerDivRequantMergePass(),
    iGELURequantMergePass(),
    iHardswishRequantMergePass(),
    PULPConvRequantMergePass(),
    MergeConstAddAndRequantPass(),
    PULPGEMMRequantMergePass(),
    PULPMatMulRequantMergePass(),
    PULPAddRequantMergePass()
])

# SCHEREMO: stdint is included before pulp_nn_kernels.h because it is supposed to be included in there, but isn't...
_includeList = [
    "pmsis.h", "stdint.h", "pulp_nn_kernels.h", "DeeployBasicMath.h", "dory_dma.h", "dory_mem.h", "bsp/ram.h"
]


class PULPClusterEngine(DeploymentEngine):

    def __init__(self, name: str, Mapping = PULPMapping, initCode = "", includeList = _includeList) -> None:
        super().__init__(name, Mapping, initCode, includeList)


class PULPPlatform(DeploymentPlatform):

    def __init__(self,
                 engines = [PULPClusterEngine("PULPCluster")],
                 variableBuffer = PULPVariableBuffer,
                 constantBuffer = PULPConstantBuffer,
                 structBuffer = PULPStructBuffer,
                 transientBuffer = PULPTransientBuffer) -> None:
        super().__init__(engines, variableBuffer, constantBuffer, structBuffer, transientBuffer)


class MemoryPULPPlatform(MemoryPlatform):

    untiledOps = ["add"]

    def __init__(self,
                 memoryHierarchy: MemoryHierarchy,
                 defaultTargetMemoryLevel: MemoryLevel,
                 engines = [PULPClusterEngine("PULPCluster")],
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


class MemoryPULPPlatformWrapper(MemoryPlatformWrapper):

    untiledOps = ["add"]

    def __init__(self, platform: PULPPlatform, memoryHierarchy: MemoryHierarchy, defaultTargetMemoryLevel: MemoryLevel):
        assert isinstance(platform, PULPPlatform), \
        f"Given platform is not an instance of PULPPlatform. Platform type: {type(platform).__name__}"
        super().__init__(platform, memoryHierarchy, defaultTargetMemoryLevel)

    def getTargetMemoryLevel(self, node: gs.Node, tensorName: str, ctxt: NetworkContext) -> str:
        if node.op in self.untiledOps:
            return ctxt.lookup(tensorName)._memoryLevel
        return super().getTargetMemoryLevel(node, tensorName, ctxt)
