from Deeploy.DeeployTypes import CodeTransformation, NodeBinding
from Deeploy.CommonExtensions.CodeTransformationPasses.MemoryAllocation import ArgumentStructGeneration, \
    MemoryManagementGeneration
from Deeploy.FutureExtension.CodeTransformationPasses.FutureCodeTransformation import FutureGeneration
from Deeploy.AbstractDataTypes import PointerClass
from Deeploy.CommonExtensions.DataTypes import FloatDataTypes, IntegerDataTypes, SignedIntegerDataTypes, float32_t, int8_t, int32_t, uint8_t
from Deeploy.Targets.Generic.TypeCheckers import GatherChecker
from Deeploy.Targets.Spatz.Templates import GatherTemplate

BasicTransformer = CodeTransformation([ArgumentStructGeneration(), MemoryManagementGeneration(), FutureGeneration()])

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