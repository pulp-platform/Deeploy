# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration, MemoryPassthroughGeneration
from Deeploy.CommonExtensions.DataTypes import float32_t, int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.MemoryLevelExtension.CodeTransformationPasses.Closure import MemoryAwareClosureGeneration
from Deeploy.Targets.Generic.Templates import ConcatTemplate, GatherTemplate, MatMulTemplate, ReshapeTemplate, \
    iNoNormTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, ConcatChecker, DivChecker, GatherChecker, GEMMChecker, \
    HardswishChecker, MatMulChecker, MulChecker, ReshapeChecker, RMSNormChecker, RQAddChecker, SoftmaxChecker, \
    TransposeChecker, iNoNormChecker
from Deeploy.Targets.Snitch.CodeTransformationPasses import SnitchClusterTiling, SnitchCoreFilterPass, \
    SnitchSynchCoresPass
from Deeploy.Targets.Snitch.DMA.SnitchDma import SnitchDma
from Deeploy.Targets.Snitch.Templates import AddTemplate, FloatGemmTemplate, FloatMatMulTemplate, RQAddTemplate, \
    TransposeTemplate, iSoftmaxTemplate
from Deeploy.Targets.Snitch.Templates.FloatAddTemplate import referenceTemplate as FloatAddTemplate
from Deeploy.Targets.Snitch.Templates.FloatDivTemplate import referenceTemplate as FloatDivTemplate
from Deeploy.Targets.Snitch.Templates.FloatHardSwishTemplate import referenceTemplate as FloatHardSwishTemplate
from Deeploy.Targets.Snitch.Templates.FloatMulTemplate import referenceTemplate as FloatMulTemplate
from Deeploy.Targets.Snitch.Templates.FloatRMSNormTemplate import ssrFrepTemplate as FloatRMSNormTemplate
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

SkipTransformer = CodeTransformation([
    SnitchSynchCoresPass(),
    ArgumentStructGeneration(),
    MemoryPassthroughGeneration("L.*"),
    MemoryPassthroughGeneration(),
    FutureGeneration()
])

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

SnitchRMSNormBindings = [
    NodeBinding(RMSNormChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatRMSNormTemplate, TiledTransformer)
]

SnitchHardSwishBindings = [
    NodeBinding(HardswishChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), FloatHardSwishTemplate,
                TiledTransformer)
]

SnitchDivBindings = [
    NodeBinding(DivChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatDivTemplate, TiledTransformer)
]

SnitchMulBindings = [
    NodeBinding(MulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                FloatMulTemplate, TiledTransformer)
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
    NodeBinding(ConcatChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
                ConcatTemplate.referenceTemplate, TiledTransformer)
]

SnitchTransposeBindings = [
    NodeBinding(TransposeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
                TransposeTemplate.referenceTemplate, TiledTransformer)
]

# Reshape Bindings (pointer passthrough, no DMA needed)
SnitchReshapeBindings = [
    NodeBinding(ReshapeChecker([PointerClass(float32_t)], [PointerClass(float32_t)]), ReshapeTemplate.referenceTemplate,
                SkipTransformer)
]

# Gather Bindings (Tiled)
SnitchGatherBindings = [
    NodeBinding(GatherChecker([PointerClass(float32_t), PointerClass(int32_t)], [PointerClass(float32_t)]),
                GatherTemplate.referenceTemplate, TiledTransformer)
]
