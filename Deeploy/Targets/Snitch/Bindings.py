# SPDX-FileCopyrightText: 2024 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from functools import partial

from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.CommonExtensions.DataTypes import float32_t, int8_t, int32_t, uint8_t
from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.MemoryLevelExtension.CodeTransformationPasses.Closure import MemoryAwareClosureGeneration
from Deeploy.Targets.Generic.Templates import iNoNormTemplate
from Deeploy.Targets.Generic.TypeCheckers import AddChecker, GEMMChecker, RQAddChecker, SoftmaxChecker, iNoNormChecker
from Deeploy.Targets.Snitch.CodeTransformationPasses import SnitchClusterTiling, SnitchCoreFilterPass, \
    SnitchSynchCoresPass
from Deeploy.Targets.Snitch.DMA.SnitchDma import SnitchDma
from Deeploy.Targets.Snitch.Templates import AddTemplate, FloatGemmTemplate, RQAddTemplate, iSoftmaxTemplate
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
