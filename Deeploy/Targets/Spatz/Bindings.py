from functools import partial

from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes, SignedIntegerDataTypes, float32_t, int8_t, int32_t
from Deeploy.Targets.Generic.TypeCheckers import GatherChecker, MatMulChecker

from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchClusterTiling import SnitchClusterTiling
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchCoreFilter import SnitchCoreFilterPass
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchClusterSynch import SnitchSynchCoresPass
from Deeploy.Targets.Spatz.DMA.SpatzDma import SpatzDma
from Deeploy.Targets.Spatz.Templates import GatherTemplate, MatMulTemplate as SpatzMatMulTemplate
from Deeploy.Targets.Generic.Templates import MatMulTemplate, FloatMatMulTemplate
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement, \
    TilingVariableReplacementUpdate

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration,
                                         closureSuffix = "_closure",
                                         startRegion = "L2",
                                         endRegion = "L1")

BasicTransformer = CodeTransformation(
    [ArgumentStructGeneration(),
    MemoryManagementGeneration(),
    FutureGeneration()])

TiledTransformer = CodeTransformation([
    SnitchCoreFilterPass("compute"),
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    SnitchSynchCoresPass(), # snrt_cluster_hw_barrier()
    TilingVariableReplacementUpdate("L1"),
    SnitchClusterTiling("L2", "L1", SpatzDma()),
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    MemoryManagementGeneration()
])

SpatzGatherBindings = [
    NodeBinding(
        GatherChecker(
            [PointerClass(type), PointerClass(int32_t)],
            [PointerClass(type)]
        ),
        GatherTemplate.referenceTemplate,
        BasicTransformer
    ) for type in SignedIntegerDataTypes] + [
    NodeBinding(
        GatherChecker(
            [PointerClass(float32_t), PointerClass(type)],
            [PointerClass(float32_t)]
        ),
        GatherTemplate.referenceTemplate,
        BasicTransformer
    ) for type in IntegerDataTypes
]

SpatzMatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                SpatzMatMulTemplate.spatzSIMatMulTemplate, TiledTransformer),
    NodeBinding(
        MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        SpatzMatMulTemplate.spatzFloatMatMulTemplate, TiledTransformer)
]
# with BEGIN_SINGLE_CORE
# SpatzMatMulBindings = [
#     NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
#                 MatMulTemplate.referenceTemplate, TiledTransformer)
# ] + [
#     NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
#                 FloatMatMulTemplate.referenceTemplate, TiledTransformer)
# ]