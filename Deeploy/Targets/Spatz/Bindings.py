from functools import partial

from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.Targets.Spatz.CodeTransformationPasses.Benchmarking import SpatzBenchmarkInnerPass, SpatzBenchmarkOuterPass

from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import IntegerDataTypes, SignedIntegerDataTypes, float32_t, int8_t, int32_t
from Deeploy.Targets.Generic.TypeCheckers import GatherChecker, MatMulChecker, TopKChecker, SoftmaxChecker

from Deeploy.CommonExtensions.CodeTransformationPasses.Closure import ClosureGeneration, MemoryAwareClosureGeneration
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchClusterTiling import SnitchClusterTiling
from Deeploy.Targets.Snitch.CodeTransformationPasses.SnitchClusterSynch import SnitchSynchCoresPass
from Deeploy.Targets.Spatz.DMA.SpatzDma import SpatzDma
from Deeploy.Targets.Spatz.Templates import GatherTemplate, MatMulTemplate as SpatzMatMulTemplate, TopKTemplate, SoftmaxTemplate
from Deeploy.Targets.Generic.Templates import MatMulTemplate, FloatMatMulTemplate
from Deeploy.TilingExtension.CodeTransformationPasses.TilingVariableReplacement import TilingVariableReplacement, \
    TilingVariableReplacementUpdate

TilingCallClosure = partial(ClosureGeneration, closureSuffix = "_tiling_closure")
MemoryAwareFunctionCallClosure = partial(MemoryAwareClosureGeneration,
                                         closureSuffix = "_closure",
                                         startRegion = "L3",
                                         endRegion = "L1")

BasicTransformer = CodeTransformation(
    [ArgumentStructGeneration(),
    MemoryManagementGeneration(),
    FutureGeneration()])

TiledTransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    TilingCallClosure(writeback = False),
    SnitchSynchCoresPass(), # snrt_cluster_hw_barrier()
    # SpatzBenchmarkInnerPass(), # <- attention: increases runtime and benchmarks only when tiling loop has one iteration
    TilingVariableReplacementUpdate("L1"),
    SnitchClusterTiling("L3", "L1", SpatzDma()),
    # SpatzBenchmarkOuterPass(), # <- attention: increases runtime and benchmarks only when tiling loop has one iteration
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    MemoryManagementGeneration("L3"),
    MemoryManagementGeneration(),
])

DynamicDMATransformer = CodeTransformation([
    TilingVariableReplacement("L1"),
    # TilingCallClosure(writeback = False),
    SnitchSynchCoresPass(), # snrt_cluster_hw_barrier()
    # SpatzBenchmarkInnerPass(), # <- attention: increases runtime and benchmarks only when tiling loop has one iteration
    TilingVariableReplacementUpdate("L1"),
    SnitchClusterTiling("L3", "L1", SpatzDma()),
    # SpatzBenchmarkOuterPass(), # <- attention: increases runtime and benchmarks only when tiling loop has one iteration
    ArgumentStructGeneration(),
    MemoryManagementGeneration("L1"),
    MemoryAwareFunctionCallClosure(writeback = False, generateStruct = True),
    MemoryManagementGeneration("L3"),
    MemoryManagementGeneration(),
])

SpatzGatherBindings = [
    NodeBinding(
        GatherChecker(
            [PointerClass(float32_t), PointerClass(type)],
            [PointerClass(float32_t)]
        ),
        GatherTemplate.dynamicDMAtemplate,
        DynamicDMATransformer
    ) for type in IntegerDataTypes
]

# SpatzGatherBindings = [
#     NodeBinding(
#         GatherChecker(
#             [PointerClass(type), PointerClass(int32_t)],
#             [PointerClass(type)]
#         ),
#         GatherTemplate.referenceTemplate,
#         BasicTransformer
#     ) for type in SignedIntegerDataTypes] +

# with tiled transformer
SpatzMatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                SpatzMatMulTemplate.spatzSIMatMulTemplate, TiledTransformer),
    NodeBinding(
        MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        SpatzMatMulTemplate.spatzFloatMatMulTemplate, TiledTransformer)
]
'''
# without tiled transformer
SpatzMatMulBindings = [
    NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
                SpatzMatMulTemplate.spatzSIMatMulTemplate, BasicTransformer),
    NodeBinding(
        MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
        SpatzMatMulTemplate.spatzFloatMatMulTemplate, BasicTransformer)
]
# with BEGIN_SINGLE_CORE
# SpatzMatMulBindings = [
#     NodeBinding(MatMulChecker([PointerClass(int8_t), PointerClass(int8_t)], [PointerClass(int32_t)]),
#                 MatMulTemplate.referenceTemplate, TiledTransformer)
# ] + [
#     NodeBinding(MatMulChecker([PointerClass(float32_t), PointerClass(float32_t)], [PointerClass(float32_t)]),
#                 FloatMatMulTemplate.referenceTemplate, TiledTransformer)
# ]
'''

SpatzTopKBindings = [
    NodeBinding(
        TopKChecker(
            [PointerClass(float32_t), PointerClass(int32_t)], # inputs
            [PointerClass(float32_t), PointerClass(int32_t)] # outputs
        ),
        TopKTemplate.SpatzTilingTemplate,
        TiledTransformer,
    )
]


SpatzSoftmaxBindings = [
    NodeBinding(
        SoftmaxChecker([PointerClass(float32_t)], [PointerClass(float32_t)]),
        SoftmaxTemplate.floatTilingTemplate,
        TiledTransformer
    )
]
# [
#     NodeBinding(
#         SoftmaxChecker([PointerClass(int8_t)], [PointerClass(int8_t)]),
#         SoftmaxTemplate.integerTilingTemplate,
#         TiledTransformer
#     )
# ]
