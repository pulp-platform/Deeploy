# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import float32_t, int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.Targets.Generic.Templates import ConcatTemplate, iNoNormTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, ConcatChecker, DivChecker, GatherChecker, GEMMChecker, \
    HardSwishChecker, MatMulChecker, MulChecker, ReshapeChecker, RMSNormChecker, RQAddChecker, SoftmaxChecker, \
    TransposeChecker, iNoNormChecker
from Deeploy.Targets.Snitch.CodeTransformationPasses import SnitchClusterTiling, SnitchCoreFilterPass, \
    SnitchSynchCoresPass
from Deeploy.Targets.Snitch.DMA.SnitchDma import SnitchDma
from Deeploy.Targets.Snitch.Templates import AddTemplate, FloatGemmTemplate, FloatMatMulTemplate, GatherTemplate, \
    MatMulTemplate, ReshapeTemplate, RQAddTemplate, TransposeTemplate, iSoftmaxTemplate
from Deeploy.Targets.Snitch.Templates.FloatAddTemplate import referenceTemplate as FloatAddTemplate
from Deeploy.Targets.Snitch.Templates.FloatDivTemplate import referenceTemplate as FloatDivTemplate
from Deeploy.Targets.Snitch.Templates.FloatHardSwishTemplate import referenceTemplate as FloatHardSwishTemplate
from Deeploy.Targets.Snitch.Templates.FloatMulTemplate import referenceTemplate as FloatMulTemplate
from Deeploy.Targets.Snitch.Templates.FloatRMSNormTemplate import referenceTemplate as FloatRMSNormTemplate
from Deeploy.Targets.Snitch.Templates.FloatSoftmaxTemplate import FloatSoftmax_Template
from Deeploy.Targets.Snitch.Templates.GemmTemplate import SnitchGemm_Template
from Deeploy.Targets.Snitch.Templates.RqGemmTemplate import SnitchRqGemm_Template
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement, \
    TilingVariableReplacementUpdate

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration,
                                         closureSuffix = "_closure",
                                         startRegion = "L2",
                                         endRegion = "L1")

BasicTransformer = CodeTransformation(
    [SnitchSynchCoresPass(),
     ArgumentStructGeneration(),
     MemoryManagementGeneration(),
     FutureGeneration()])

TiledTransformer = CodeTransformation([
    SnitchCoreFilterPass("compute"),
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    SnitchSynchCoresPass(),
    TilingVariableReplacementUpdate("L1"),
    SnitchClusterTiling("L2", "L1", SnitchDma()),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    MemoryManagementGeneration("L2"),
    MemoryManagementGeneration()
])

SnitchiSoftmaxBindings = [
    NodeBinding(SoftmaxChecker([PointerClass(_type)], [PointerClass(uint8_t)]), iSoftmaxTemplate.referenceTemplate,
                TiledTransformer) for _type in [int8_t, uint8_t]
] + [
    NodeBinding(SoftmaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatSoftmax_Template,
                TiledTransformer)
]

SnitchiNoNormBindings = [
    NodeBinding(
        iNoNormChecker([PointerClass(_type), PointerClass(int8_t),
                        PointerClass(int32_t)], [PointerClass(int8_t)]), iNoNormTemplate.referenceTemplate,
        TiledTransformer) for _type in [int8_t]
]
SnitchRQAddBindings = [
    NodeBinding(RQAddChecker([PointerClass(_type), PointerClass(_type)], [PointerClass(_type)]),
                RQAddTemplate.referenceTemplate, TiledTransformer) for _type in [int8_t]
]
SnitchAddBindings = [
    NodeBinding(AddChecker([PointerClass(_type), PointerClass(_type)], [PointerClass(int32_t)]),
                AddTemplate.referenceTemplate, TiledTransformer) for _type in [int8_t]
] + [
    # fp32 support
    NodeBinding(AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAddTemplate, TiledTransformer)
]

# Basic (non-tiled) FP32 Add Bindings
BasicAddBindings = [
    NodeBinding(AddChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatAddTemplate, BasicTransformer)
]

SnitchGemmBindings = [
    NodeBinding(
        GEMMChecker([PointerClass(int8_t), PointerClass(int8_t),
                     PointerClass(int32_t)], [PointerClass(int32_t)]), SnitchGemm_Template, TiledTransformer)
] + [
    NodeBinding(
        GEMMChecker([PointerClass(float32_t), PointerClass(float32_t),
                     PointerClass(float32_t)], [PointerClass(float32_t)]), FloatGemmTemplate.referenceTemplate,
        TiledTransformer)
]
SnitchRqGemmBindings = [
    NodeBinding(
        GEMMChecker([
            PointerClass(int8_t),
            PointerClass(int8_t),
            PointerClass(int32_t),
            PointerClass(int32_t),
            PointerClass(int32_t)
        ], [PointerClass(int8_t)]), SnitchRqGemm_Template, TiledTransformer)
]

# RMSNorm Bindings (Tiled)
SnitchRMSNormBindings = [
    NodeBinding(RMSNormChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatRMSNormTemplate, TiledTransformer)
]

# RMSNorm Bindings (Non-tiled)
BasicRMSNormBindings = [
    NodeBinding(RMSNormChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatRMSNormTemplate, BasicTransformer)
]

# HardSwish Bindings (Tiled)
SnitchHardSwishBindings = [
    NodeBinding(HardSwishChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatHardSwishTemplate,
                TiledTransformer)
]

# HardSwish Bindings (Non-tiled)
BasicHardSwishBindings = [
    NodeBinding(HardSwishChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatHardSwishTemplate,
                BasicTransformer)
]

# Div Bindings (Tiled)
SnitchDivBindings = [
    NodeBinding(DivChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatDivTemplate, TiledTransformer)
]

# Div Bindings (Non-tiled)
BasicDivBindings = [
    NodeBinding(DivChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatDivTemplate, BasicTransformer)
]

# Mul Bindings (Tiled)
SnitchMulBindings = [
    NodeBinding(MulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMulTemplate, TiledTransformer)
]

# Mul Bindings (Non-tiled)
BasicMulBindings = [
    NodeBinding(MulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMulTemplate, BasicTransformer)
]

# MatMul Bindings (Tiled)
SnitchMatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                MatMulTemplate.referenceTemplate, TiledTransformer),
    NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMatMulTemplate.referenceTemplate, TiledTransformer)
]

# Concat Bindings (Tiled)
SnitchConcatBindings = [
    NodeBinding(ConcatChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int8_t)]),
                ConcatTemplate.referenceTemplate, TiledTransformer),
    NodeBinding(ConcatChecker([PointerClass(int32_t), PointerClass(int32_t)], [PointerClass(int32_t)]),
                ConcatTemplate.referenceTemplate, TiledTransformer),
    NodeBinding(ConcatChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                ConcatTemplate.referenceTemplate, TiledTransformer)
]

# Transpose Bindings (Tiled)
SnitchTransposeBindings = [
    NodeBinding(TransposeChecker([PointerClass(int8_t)], [PointerClass(int8_t)]), TransposeTemplate.referenceTemplate,
                TiledTransformer),
    NodeBinding(TransposeChecker([PointerClass(int32_t)], [PointerClass(int32_t)]), TransposeTemplate.referenceTemplate,
                TiledTransformer),
    NodeBinding(TransposeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                TransposeTemplate.referenceTemplate, TiledTransformer)
]

# Transpose Bindings (Non-tiled, multi-core)
BasicSnitchTransposeBindings = [
    NodeBinding(TransposeChecker([PointerClass(int8_t)], [PointerClass(int8_t)]), TransposeTemplate.referenceTemplate,
                BasicTransformer),
    NodeBinding(TransposeChecker([PointerClass(int32_t)], [PointerClass(int32_t)]), TransposeTemplate.referenceTemplate,
                BasicTransformer),
    NodeBinding(TransposeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                TransposeTemplate.referenceTemplate, BasicTransformer)
]

# Reshape Bindings (Tiled)
SnitchReshapeBindings = [
    NodeBinding(ReshapeChecker([PointerClass(int8_t)], [PointerClass(int8_t)]), ReshapeTemplate.referenceTemplate,
                TiledTransformer),
    NodeBinding(ReshapeChecker([PointerClass(int32_t)], [PointerClass(int32_t)]), ReshapeTemplate.referenceTemplate,
                TiledTransformer),
    NodeBinding(ReshapeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), ReshapeTemplate.referenceTemplate,
                TiledTransformer)
]

# Gather Bindings (Tiled)
SnitchGatherBindings = [
    NodeBinding(GatherChecker([PointerClass(int8_t), PointerClass(int32_t)], [PointerClass(int8_t)]),
                GatherTemplate.referenceTemplate, TiledTransformer),
    NodeBinding(GatherChecker([PointerClass(int32_t), PointerClass(int32_t)], [PointerClass(int32_t)]),
                GatherTemplate.referenceTemplate, TiledTransformer),
    NodeBinding(GatherChecker([PointerClass(float32_t), PointerClass(int32_t)], [PointerClass(float32_t)]),
                GatherTemplate.referenceTemplate, TiledTransformer)
]
